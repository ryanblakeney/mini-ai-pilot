package miniai.pilot

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import java.io.BufferedReader
import java.io.File
import java.io.FileWriter
import java.io.InputStreamReader
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket
import kotlin.system.exitProcess

/*
 * TCP server: the sim bridge on the Windows laptop dials IN across tailscale.
 * Line-delimited UTF-8 JSON, one message per line, single connected client
 * (accept the first; after it disconnects, wait for the next).
 *
 * IN  (bridge -> runner):
 *   {"t":"hello",...}
 *   {"t":"ep_start","ep":<int>,"seed":<int>}          -> gate.reset()
 *   {"t":"obs","ep":<int>,"step":<int>,"obs":[15 floats],"pos_err":<float>}
 *   {"t":"ep_end",...}
 *   {"t":"run_end",...}
 * OUT (runner -> bridge, after each obs):
 *   {"t":"act","a":[ax,ay,az],"cause":"<safety cause>"}   (a = safety-FILTERED action)
 *
 * JSONL log per obs:
 *   {"ep":0,"step":3,"raw":[...],"action":[...],"cause":"pass","telemetry_age":0.55}
 * plus {"ev":"ep_start",...}, {"ev":"ep_end"}, {"ev":"run_end"}, {"ev":"client_lost"}.
 */

private const val USAGE = """
usage: LoopbackRunner [--model <path>] [--backend onnx|tflite|auto] [--listen host:port]
                      [--max-delta <float>] [--box <float>] [--log <path>] [--selftest]
  --model      default ../export/sac_actor_hover_v7.onnx (relative to pilot/phone)
  --backend    default auto (prefers onnx)
  --listen     default 0.0.0.0:9931
  --max-delta  SafetyGate maxDeltaPerStep override
  --box        SafetyGate hoverBoxRadius override
  --log        JSONL log path (default: none)
  --selftest   load model, verify golden obs/actions, gate smoke test, exit 0/1
"""

private val mapper: ObjectMapper = jacksonObjectMapper()

fun main(args: Array<String>) {
    var modelPath = "../export/sac_actor_hover_v7.onnx"
    var backend = "auto"
    var host = "0.0.0.0"
    var port = 9931
    var maxDelta: Float? = null
    var box: Float? = null
    var logPath: String? = null
    var selftest = false

    var i = 0
    while (i < args.size) {
        when (args[i]) {
            "--model" -> { i++; modelPath = args.getOrElse(i) { usageFail() } }
            "--backend" -> { i++; backend = args.getOrElse(i) { usageFail() } }
            "--listen" -> {
                i++
                val v = args.getOrElse(i) { usageFail() }
                val c = v.lastIndexOf(':')
                require(c > 0 && c < v.length - 1) { "bad --listen '$v' (want host:port)" }
                host = v.substring(0, c); port = v.substring(c + 1).toIntOrNull() ?: usageFail()
            }
            "--max-delta" -> { i++; maxDelta = args.getOrElse(i) { usageFail() }.toFloatOrNull() ?: usageFail() }
            "--box" -> { i++; box = args.getOrElse(i) { usageFail() }.toFloatOrNull() ?: usageFail() }
            "--log" -> { i++; logPath = args.getOrElse(i) { usageFail() } }
            "--selftest" -> selftest = true
            "--help", "-h" -> { println(USAGE); return }
            else -> { System.err.println("unknown arg: ${args[i]}"); usageFail() }
        }
        i++
    }

    val cfg = SafetyConfig(
        hoverBoxRadius = box ?: SafetyConfig().hoverBoxRadius,
        maxDeltaPerStep = maxDelta ?: SafetyConfig().maxDeltaPerStep,
    )

    if (selftest) exitProcess(runSelfTest(modelPath, backend, cfg))

    val model = PilotModel.open(modelPath, backend)
    val gate = SafetyGate(cfg)
    val logW = logPath?.let { java.io.PrintWriter(FileWriter(File(it), true), true) }
    ServerSocket().use { server ->
        server.reuseAddress = true
        server.bind(InetSocketAddress(host, port))
        println("pilot-loopback listening on $host:$port backend=${model.backend} model=$modelPath")
        while (true) {
            val sock = server.accept()
            println("client connected: ${sock.remoteSocketAddress}")
            try {
                handleClient(sock, model, gate, logW)
            } catch (e: Exception) {
                System.err.println("client session ended: $e")
            }
            try { sock.close() } catch (_: Exception) { }
            jsonl(logW, linkedMapOf("ev" to "client_lost"))
        }
    }
}

private fun usageFail(): Nothing {
    System.err.println(USAGE)
    exitProcess(2)
}

private fun handleClient(sock: Socket, model: PilotModel, gate: SafetyGate, logW: java.io.PrintWriter?) {
    sock.tcpNoDelay = true
    val reader = BufferedReader(InputStreamReader(sock.getInputStream(), Charsets.UTF_8))
    val out = sock.getOutputStream()
    var prevAction: FloatArray? = null
    var lastObsNanos: Long? = null

    while (true) {
        val line = reader.readLine() ?: break
        if (line.isBlank()) continue
        val msg = try {
            mapper.readTree(line)
        } catch (e: Exception) {
            System.err.println("skipping malformed line: $line")
            continue
        }
        when (msg.path("t").asText()) {
            "hello" -> jsonl(logW, linkedMapOf("ev" to "hello", "detail" to msg.toString()))

            "ep_start" -> {
                gate.reset()
                prevAction = null
                lastObsNanos = null
                jsonl(
                    logW,
                    linkedMapOf("ev" to "ep_start", "ep" to msg.path("ep").asInt(-1), "seed" to msg.path("seed").asInt(-1)),
                )
            }

            "obs" -> {
                val obsNode = msg.path("obs")
                if (obsNode.size() != PilotModel.OBS_DIM || !obsNode.isArray) {
                    System.err.println("obs message must carry ${PilotModel.OBS_DIM} floats, got: $line")
                    continue
                }
                val obs = FloatArray(PilotModel.OBS_DIM) { obsNode.get(it).floatValue() }
                val now = System.nanoTime()
                val ageSec = lastObsNanos?.let { (now - it) / 1_000_000_000.0f } ?: 0f
                lastObsNanos = now

                val raw = model.predict(obs)
                // batteryPct and DepthGrid are null in loopback; the filtered action is
                // what gets fed back as prevAction on the next step.
                val dec = gate.filter(obs, prevAction, null, ageSec, null, raw)
                prevAction = dec.action

                val resp = LinkedHashMap<String, Any>()
                resp["t"] = "act"
                resp["a"] = dec.action.roundList()
                resp["cause"] = dec.cause
                out.write((mapper.writeValueAsString(resp) + "\n").toByteArray(Charsets.UTF_8))
                out.flush()

                val rec = LinkedHashMap<String, Any>()
                rec["ep"] = msg.path("ep").asInt(-1)
                rec["step"] = msg.path("step").asInt(-1)
                rec["raw"] = raw.roundList()
                rec["action"] = dec.action.roundList()
                rec["cause"] = dec.cause
                rec["telemetry_age"] = (ageSec * 10000f).toLong() / 10000.0
                jsonl(logW, rec)
            }

            "ep_end", "run_end" ->
                jsonl(logW, linkedMapOf("ev" to msg.path("t").asText(), "ep" to msg.path("ep").asInt(-1)))

            else -> Unit // unknown message types are ignored
        }
    }
}

private fun jsonl(w: java.io.PrintWriter?, rec: Map<String, Any>) {
    w?.println(mapper.writeValueAsString(rec))
}

/** Serialize floats without double-widening noise (6 decimals is >> m/s precision). */
private fun FloatArray.roundList(): List<Double> = map { (it * 1e6).let { v -> if (v >= 0) (v + 0.5).toLong() else (v - 0.5).toLong() } / 1e6 }

// ---------------------------------------------------------------- selftest

private fun runSelfTest(modelPath: String, backend: String, cfg: SafetyConfig): Int {
    println("=== pilot selftest ===")
    println("model=$modelPath backend=$backend")
    var ok = true

    val model = try {
        PilotModel.open(modelPath, backend)
    } catch (e: Exception) {
        println("BACKEND ${backend}: FAIL to open — ${e.message}")
        return 1
    }
    println("BACKEND ${model.backend}: opened ${File(modelPath).name}")

    // Golden cases: obs rows + exact float32 ONNX actions generated from the venv.
    val golden = try {
        object {}.javaClass.getResourceAsStream("/golden_obs.json")?.use { mapper.readTree(it) }
            ?: run { println("GOLDEN: FAIL — resource /golden_obs.json missing"); return 1 }
    } catch (e: Exception) {
        println("GOLDEN: FAIL reading resource — ${e.message}"); return 1
    }
    val obsArr = golden.path("obs"); val actArr = golden.path("actions")
    val n = obsArr.size()
    var maxDiff = 0f
    var worst = -1
    for (k in 0 until n) {
        val row = FloatArray(obsArr.get(k).size()) { obsArr.get(k).get(it).floatValue() }
        val expected = actArr.get(k)
        val got = model.predict(row)
        for (j in 0 until PilotModel.ACT_DIM) {
            val d = kotlin.math.abs(got[j] - expected.get(j).floatValue())
            if (d > maxDiff) { maxDiff = d; worst = k }
        }
    }
    val tol = if (model.backend == "onnx") 1e-5f else 1e-4f
    println(
        "GOLDEN ${model.backend}: rows=$n max_abs_diff=%.3e (worst row %d) tol=%.0e -> %s"
            .format(maxDiff, worst, tol, if (maxDiff < tol) "PASS" else "FAIL"),
    )
    if (!(maxDiff < tol)) ok = false
    model.close()

    // SafetyGate smoke: normal in-box obs with the policy's own golden action -> pass.
    val gate = SafetyGate(cfg)
    val firstObs = FloatArray(obsArr.get(0).size()) { obsArr.get(0).get(it).floatValue() }
    val smoke = gate.filter(firstObs, null, 100f, 0f, null)
    val smokeOk = smoke.cause.startsWith("pass")
    println("GATE smoke: cause=${smoke.cause} action=${smoke.action.toList()} -> ${if (smokeOk) "PASS" else "FAIL"}")
    if (!smokeOk) ok = false

    println("=== selftest ${if (ok) "PASS" else "FAIL"} ===")
    return if (ok) 0 else 1
}
