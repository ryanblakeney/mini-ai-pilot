package miniai.pilot

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Tunables for [SafetyGate]. All distances meters, speeds m/s, angles radians,
 * battery percent (0..100), times seconds.
 */
data class SafetyConfig(
    /** norm(position error) beyond this -> P-pull to hover target. */
    val hoverBoxRadius: Float = 6.0f,
    /** norm(position error) beyond this -> full-speed pull to target. */
    val geofenceRadius: Float = 7.5f,
    /** Per-axis commanded-speed cap (m/s). */
    val maxSpeed: Float = 1.0f,
    /** Per-axis maximum change between consecutive commands (m/s per step). */
    val maxDeltaPerStep: Float = 0.5f,
    /** Age of last fresh telemetry above which we hover in place. */
    val telemetryLossSec: Float = 2.0f,
    /** Battery below this (%) -> hover in place + flag [SafetyGate.returnNeeded]. */
    val batteryReturnPct: Float = 30f,
    /** Battery below this (%) -> forced descend (0, 0, +1) (NED: z is DOWN). */
    val batteryLandPct: Float = 15f,
    /** Minimum allowed clearance (m) in the heading of motion before a veto. */
    val depthVetoMeters: Float = 1.0f,
)

/**
 * Minimal depth sensor summary: distance (m) to the nearest obstacle and the
 * XY (North, East) *unit* vector from the drone toward that obstacle.
 */
class DepthGrid(val nearestDistanceM: Float, val nearestUnitXY: FloatArray)

/** A gated command: [action] = [v_north, v_east, v_down] m/s + human-readable [cause]. */
data class Decision(val action: FloatArray, val cause: String)

/**
 * Pure-logic safety filter for the SAC hover policy. No I/O, no Android deps,
 * allocation-light. Stateless except [returnNeeded]: the previous command is
 * always supplied by the caller as [filter]'s `prevAction` and never cached.
 *
 * obs layout (float32, NED, z DOWN), length 15:
 *  [0..2]  position error vs hover target (m)
 *  [3..5]  linear velocity (m/s)
 *  [6..8]  roll, pitch, yaw (rad)
 *  [9..11] angular velocity (rad/s)
 *  [12..14] last action (velocity setpoints m/s)
 *
 * Evaluation order (first match wins):
 *  1. telemetry loss          -> zeros,            "telemetry_loss_hover"
 *  2. battery <= land         -> (0,0,+1),         "battery_land"
 *  3. battery <= return       -> zeros,            "battery_return_hover" (sets [returnNeeded])
 *  4. norm(err) >= geofence   -> clipped P-pull,   "geofence_pullin"
 *  5. norm(err) >= hover box  -> clipped P-pull,   "hover_box_pullin"
 *  6. obstacle close + moving toward it -> zeros,  "obstacle_veto"
 *  7. policy action, rate-limited vs prevAction and speed-clamped ->
 *     "pass" [|rate_limited] [|speed_capped]
 *
 * The policy action is the optional 6th parameter. When omitted it is taken
 * from obs[12..14] (the last-applied command), which keeps the 5-parameter
 * call shape specified for the gate usable stand-alone. The loopback runner
 * always passes the freshly inferred policy output explicitly.
 */
class SafetyGate(
    val config: SafetyConfig = SafetyConfig(),
    val nanoTime: () -> Long = System::nanoTime,
) {
    /** Set true when the battery-return rule fires. Cleared by [reset]. */
    @Volatile
    var returnNeeded: Boolean = false
        private set

    /** Clears gate state (the [returnNeeded] flag; prevAction is caller-supplied, never cached). */
    fun reset() {
        returnNeeded = false
    }

    fun filter(
        obs: FloatArray,
        prevAction: FloatArray?,
        batteryPct: Float?,
        telemetryAgeSec: Float,
        depthGrid: DepthGrid?,
        policyAction: FloatArray? = null,
    ): Decision {
        val cfg = config

        // 1. Telemetry loss beats everything.
        if (telemetryAgeSec > cfg.telemetryLossSec) {
            return Decision(FloatArray(3), "telemetry_loss_hover")
        }

        // 2 / 3. Battery rules beat geofence/hover-box/obstacle.
        if (batteryPct != null) {
            if (batteryPct <= cfg.batteryLandPct) {
                return Decision(floatArrayOf(0f, 0f, 1f), "battery_land")
            }
            if (batteryPct <= cfg.batteryReturnPct) {
                returnNeeded = true
                return Decision(FloatArray(3), "battery_return_hover")
            }
        }

        // 4 / 5. Geofence and hover box: full-strength P toward the target.
        val ex = obs[0]
        val ey = obs[1]
        val ez = obs[2]
        val dist = sqrt(ex * ex + ey * ey + ez * ez)
        if (dist >= cfg.geofenceRadius) {
            return Decision(
                floatArrayOf(clip(-ex, cfg.maxSpeed), clip(-ey, cfg.maxSpeed), clip(-ez, cfg.maxSpeed)),
                "geofence_pullin",
            )
        }
        if (dist >= cfg.hoverBoxRadius) {
            return Decision(
                floatArrayOf(clip(-ex, cfg.maxSpeed), clip(-ey, cfg.maxSpeed), clip(-ez, cfg.maxSpeed)),
                "hover_box_pullin",
            )
        }

        val policy = policyAction ?: floatArrayOf(obs[12], obs[13], obs[14])

        // 6. Obstacle veto: close AND the policy command moves toward it.
        if (depthGrid != null && depthGrid.nearestDistanceM < cfg.depthVetoMeters) {
            val u = depthGrid.nearestUnitXY
            val dot = policy[0] * u[0] + policy[1] * u[1] // unit extended to 3D as (x, y, 0)
            if (dot > 0f) {
                return Decision(FloatArray(3), "obstacle_veto")
            }
        }

        // 7. Pass: rate-limit against prevAction, then clamp per-axis speed.
        var rateLimited = false
        var speedCapped = false
        val out = FloatArray(3)
        for (i in 0..2) {
            var a = policy[i]
            if (prevAction != null) {
                val lo = prevAction[i] - cfg.maxDeltaPerStep
                val hi = prevAction[i] + cfg.maxDeltaPerStep
                val rl = if (a < lo) lo else if (a > hi) hi else a
                if (abs(rl - a) > 1e-6f) rateLimited = true
                a = rl
            }
            val cl = clip(a, cfg.maxSpeed)
            if (abs(cl - a) > 1e-6f) speedCapped = true
            out[i] = cl
        }
        val cause = buildString(24) {
            append("pass")
            if (rateLimited) append("|rate_limited")
            if (speedCapped) append("|speed_capped")
        }
        return Decision(out, cause)
    }

    private fun clip(v: Float, limit: Float): Float =
        if (v > limit) limit else if (v < -limit) -limit else v
}
