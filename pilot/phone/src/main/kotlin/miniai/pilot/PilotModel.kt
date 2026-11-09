package miniai.pilot

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * SAC actor inference over the exported hover policy.
 * Input "obs" [1,15] float32 -> output "action" [1,3] float32 in [-1,1].
 * Deterministic (tanh(mu)) — never sample.
 *
 * Backends:
 *  - "onnx": ONNX Runtime Java (desktop-capable, the loopback default).
 *  - "tflite": best-effort desktop TFLite. No published TFLite Java artifact is
 *    desktop-capable (see [TFLITE_UNAVAILABLE]); resolved via reflection so the
 *    build stays green without any TFLite dependency.
 *  - "auto": prefer onnx.
 */
class PilotModel private constructor(
    val backend: String,
    private val predictFn: (FloatArray) -> FloatArray,
    private val closeFn: () -> Unit,
) {
    /** Run one [1,15] observation through the policy, return 3 floats. */
    fun predict(obs: FloatArray): FloatArray = predictFn(obs)

    fun close() = closeFn()

    companion object {
        const val OBS_DIM = 15
        const val ACT_DIM = 3

        const val TFLITE_UNAVAILABLE: String =
            "No desktop-capable TFLite Java runtime found on the classpath. " +
                "Tried: org.tensorflow:tensorflow-lite 2.16.1 and 2.17.0 (Maven Central — packaging=aar, Android-only), " +
                "com.google.ai.edge.litert:litert 1.0.0 .. 2.3.2 (Google Maven dl.google.com — packaging=aar, Android-only). " +
                "Class org.tensorflow.lite.Interpreter is not loadable on this JVM. " +
                "Use --backend onnx (the loopback backend)."

        fun open(path: String, backend: String = "auto"): PilotModel = when (backend.lowercase()) {
            "onnx" -> openOnnx(path)
            "tflite" -> openTflite(path)
            "auto" -> try {
                openOnnx(path)
            } catch (e: Exception) {
                val t = openTfliteOrNull(path)
                if (t != null) t else throw IllegalStateException(
                    "backend=auto: ONNX failed ($e) and no TFLite desktop runtime is available. $TFLITE_UNAVAILABLE", e,
                )
            }
            else -> throw IllegalArgumentException("unknown backend '$backend' (expected onnx|tflite|auto)")
        }

        private fun openOnnx(path: String): PilotModel {
            val f = File(path)
            require(f.isFile) { "model file not found: $f.absolutePath" }
            val env = OrtEnvironment.getEnvironment()
            val session = env.createSession(f.absolutePath)
            val inName = session.inputNames.first()   // "obs"
            val outName = session.outputNames.first() // "action"
            return PilotModel(
                backend = "onnx",
                predictFn = { obs ->
                    require(obs.size == OBS_DIM) { "obs must have $OBS_DIM floats, got ${obs.size}" }
                    val fb = ByteBuffer.allocateDirect(OBS_DIM * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
                    fb.put(obs).rewind()
                    OnnxTensor.createTensor(env, fb, longArrayOf(1, OBS_DIM.toLong())).use { input ->
                        session.run(mapOf(inName to input)).use { result ->
                            val t = result.get(outName).get() as OnnxTensor
                            val ofb = t.floatBuffer
                            FloatArray(ACT_DIM) { ofb.get(it) }
                        }
                    }
                },
                closeFn = { session.close() },
            )
        }

        /** Reflective TFLite: only usable if some desktop jar put org.tensorflow.lite.Interpreter on the classpath. */
        private fun openTfliteOrNull(path: String): PilotModel? = try {
            openTflite(path)
        } catch (e: Exception) {
            null
        }

        private fun openTflite(path: String): PilotModel {
            val f = File(path)
            require(f.isFile) { "model file not found: $f.absolutePath" }
            val interpCls = try {
                Class.forName("org.tensorflow.lite.Interpreter")
            } catch (t: Throwable) {
                throw UnsupportedOperationException(TFLITE_UNAVAILABLE, t)
            }
            try {
                val modelBuf = FileChannel.open(f.toPath()).use { ch ->
                    ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size()) as ByteBuffer
                }
                val interp = interpCls.getConstructor(ByteBuffer::class.java).newInstance(modelBuf)
                val tensorCls = Class.forName("org.tensorflow.lite.Tensor")
                val allocateBuf = tensorCls.getMethod("allocateTensorBuffer")
                val inTensor = interpCls.getMethod("getInputTensor", Int::class.javaPrimitiveType).invoke(interp, 0)
                val outTensor = interpCls.getMethod("getOutputTensor", Int::class.javaPrimitiveType).invoke(interp, 0)
                val runM = interpCls.getMethod("run", Any::class.java, Any::class.java)
                val inBuf = allocateBuf.invoke(inTensor) as ByteBuffer
                val outBuf = allocateBuf.invoke(outTensor) as ByteBuffer
                inBuf.order(ByteOrder.nativeOrder())
                outBuf.order(ByteOrder.nativeOrder())
                return PilotModel(
                    backend = "tflite",
                    predictFn = { obs ->
                        require(obs.size == OBS_DIM) { "obs must have $OBS_DIM floats, got ${obs.size}" }
                        inBuf.clear(); inBuf.asFloatBuffer().put(obs); inBuf.position(OBS_DIM * 4)
                        outBuf.clear()
                        runM.invoke(interp, inBuf, outBuf)
                        val fb = outBuf.asFloatBuffer()
                        FloatArray(ACT_DIM) { fb.get(it) }
                    },
                    closeFn = { interpCls.getMethod("close").invoke(interp) },
                )
            } catch (e: Exception) {
                throw UnsupportedOperationException(
                    "org.tensorflow.lite.Interpreter was found but failed to initialize: $e. $TFLITE_UNAVAILABLE", e,
                )
            }
        }
    }
}
