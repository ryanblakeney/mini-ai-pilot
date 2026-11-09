import miniai.pilot.Decision
import miniai.pilot.DepthGrid
import miniai.pilot.SafetyConfig
import miniai.pilot.SafetyGate
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertIterableEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import kotlin.math.abs

/**
 * obs is always length 15:
 *  [0..2] pos err (N,E,D) m | [3..5] vel | [6..8] rpy | [9..11] ang vel | [12..14] last action
 * Rule-7 tests put the candidate policy command in obs[12..14] (the 5-parameter
 * call shape) or pass policyAction explicitly (6th parameter).
 */
class SafetyGateTest {

    private val gate = SafetyGate()
    private val cfg = SafetyConfig()

    private fun obs(
        posErr: FloatArray = floatArrayOf(0f, 0f, 0f),
        lastAction: FloatArray = floatArrayOf(0f, 0f, 0f),
    ): FloatArray {
        val o = FloatArray(15)
        o[0] = posErr[0]; o[1] = posErr[1]; o[2] = posErr[2]
        o[12] = lastAction[0]; o[13] = lastAction[1]; o[14] = lastAction[2]
        return o
    }

    private fun assertAction(expected: FloatArray, actual: FloatArray, msg: String = "") {
        assertEquals(expected.size, actual.size, msg)
        for (i in expected.indices) {
            assertTrue(abs(expected[i] - actual[i]) < 1e-6f, "$msg axis $i: expected ${expected[i]} got ${actual[i]}")
        }
    }

    // ---------------- rule 1: telemetry loss ----------------

    @Test
    fun `telemetry loss hovers`() {
        val d = gate.filter(obs(), null, 100f, 2.001f, null)
        assertEquals("telemetry_loss_hover", d.cause)
        assertAction(floatArrayOf(0f, 0f, 0f), d.action)
    }

    @Test
    fun `telemetry boundary not exceeded passes`() {
        // age exactly == telemetryLossSec is NOT loss (rule is `>`)
        val d = gate.filter(obs(), null, 100f, 2.0f, null)
        assertTrue(d.cause.startsWith("pass"), "cause=${d.cause}")
    }

    @Test
    fun `telemetry loss beats battery`() {
        val d = gate.filter(obs(), null, 5f, 5f, null)
        assertEquals("telemetry_loss_hover", d.cause)
    }

    @Test
    fun `telemetry loss beats geofence`() {
        val d = gate.filter(obs(posErr = floatArrayOf(20f, 0f, 0f)), null, 100f, 5f, null)
        assertEquals("telemetry_loss_hover", d.cause)
    }

    // ---------------- rules 2-3: battery ----------------

    @Test
    fun `battery land at boundary`() {
        val d = gate.filter(obs(), null, 15f, 0f, null)
        assertEquals("battery_land", d.cause)
        assertAction(floatArrayOf(0f, 0f, 1f), d.action)
    }

    @Test
    fun `battery land below threshold`() {
        val d = gate.filter(obs(), null, 3f, 0f, null)
        assertEquals("battery_land", d.cause)
    }

    @Test
    fun `battery return hover at boundary and sets flag`() {
        val g = SafetyGate()
        assertFalse(g.returnNeeded)
        val d = g.filter(obs(), null, 30f, 0f, null)
        assertEquals("battery_return_hover", d.cause)
        assertAction(floatArrayOf(0f, 0f, 0f), d.action)
        assertTrue(g.returnNeeded)
    }

    @Test
    fun `battery return not triggered above threshold`() {
        val g = SafetyGate()
        g.filter(obs(), null, 30.001f, 0f, null)
        assertFalse(g.returnNeeded)
    }

    @Test
    fun `battery null keeps land and return rules inert`() {
        val d = gate.filter(obs(), null, null, 0f, null)
        assertTrue(d.cause.startsWith("pass"), d.cause)
    }

    @Test
    fun `battery beats geofence`() {
        val d = gate.filter(obs(posErr = floatArrayOf(9f, 0f, 0f)), null, 10f, 0f, null)
        assertEquals("battery_land", d.cause)
        val d2 = gate.filter(obs(posErr = floatArrayOf(9f, 0f, 0f)), null, 20f, 0f, null)
        assertEquals("battery_return_hover", d2.cause)
    }

    @Test
    fun `land priority over return when both match`() {
        // 10f <= 15 (land) and <= 30 (return): land must win (rule order).
        val d = gate.filter(obs(), null, 10f, 0f, null)
        assertEquals("battery_land", d.cause)
    }

    // ---------------- rules 4-5: geofence / hover box ----------------

    @Test
    fun `geofence triggers at exact boundary and pulls at full speed`() {
        // norm = 7.5 exactly -> >= triggers
        val d = gate.filter(obs(posErr = floatArrayOf(7.5f, 0f, 0f)), null, 100f, 0f, null)
        assertEquals("geofence_pullin", d.cause)
        assertAction(floatArrayOf(-1f, 0f, 0f), d.action) // clipped to -maxSpeed
    }

    @Test
    fun `geofence clips each axis of P-pull`() {
        // norm(6,0,4.5) = 7.5 >= 7.5; per-axis clip(-obs, +-1)
        val d = gate.filter(obs(posErr = floatArrayOf(6f, 0f, 4.5f)), null, 100f, 0f, null)
        assertEquals("geofence_pullin", d.cause)
        assertAction(floatArrayOf(-1f, 0f, -1f), d.action)
    }

    @Test
    fun `geofence P-pull unscaled inside speed cap`() {
        // err inside geofence per-axis: small axis components keep proportional P value
        val g = SafetyGate(SafetyConfig(geofenceRadius = 5f, maxSpeed = 1f))
        val d = g.filter(obs(posErr = floatArrayOf(5f, 0.3f, 0f)), null, 100f, 0f, null) // norm > 5
        assertEquals("geofence_pullin", d.cause)
        assertAction(floatArrayOf(-1f, -0.3f, 0f), d.action)
    }

    @Test
    fun `geofence beats hover box`() {
        val d = gate.filter(obs(posErr = floatArrayOf(8f, 0f, 0f)), null, 100f, 0f, null)
        assertEquals("geofence_pullin", d.cause)
    }

    @Test
    fun `hover box triggers at exact boundary`() {
        val d = gate.filter(obs(posErr = floatArrayOf(6f, 0f, 0f)), null, 100f, 0f, null)
        assertEquals("hover_box_pullin", d.cause)
        assertAction(floatArrayOf(-1f, 0f, 0f), d.action)
    }

    @Test
    fun `hover box clips pull per-axis`() {
        // norm(3,5,4) = 7.07 -> inside geofence? sqrt(50)=7.07 < 7.5 yes; >= 6 box
        val d = gate.filter(obs(posErr = floatArrayOf(3f, 5f, 4f)), null, 100f, 0f, null)
        assertEquals("hover_box_pullin", d.cause)
        assertAction(floatArrayOf(-1f, -1f, -1f), d.action)
    }

    @Test
    fun `hover box P-pull proportional below cap`() {
        // norm(1.2, 5.8, 0)=5.924 < 6? no... use 5.9: norm(1.2,5.65,0)=5.77 hmm - use z-axis: norm(0,0,6.3) no cap? 6.3>1 clips. Use small err with big axis err but low norm is impossible >=6 & |err|<1 on triggered axis only via 3 axes: (2,5.5,0): norm=5.85 <6. (2,5.8,0): 6.13 >= 6 -> pull (-2,-1(clip),0)? -5.8 clips to -1. Use (0.8,5.75,0.8): norm=5.875 no. (1,5.8,1): norm=6.045>=6 -> (-1(clip),-1,-1) all clip 1? -5.8 clips, 1 fine: (-1,-1,-1). proportional check: (0.5,5.5,2.1): norm=5.98 no. (1.5,5.4,2.4): norm=6.15 -> (-1clip, -1clip, -1clip)... choose (0.4,5.6,2.3): 6.10 -> axis x clip(-0.4)=-0.4 proportional survives.
        val d = gate.filter(obs(posErr = floatArrayOf(0.4f, 5.6f, 2.3f)), null, 100f, 0f, null)
        assertEquals("hover_box_pullin", d.cause)
        assertAction(floatArrayOf(-0.4f, -1f, -1f), d.action)
    }

    @Test
    fun `hover box beats obstacle veto`() {
        val veto = DepthGrid(0.5f, floatArrayOf(1f, 0f))
        val d = gate.filter(obs(posErr = floatArrayOf(6.5f, 0f, 0f)), null, 100f, 0f, veto)
        assertEquals("hover_box_pullin", d.cause)
    }

    // ---------------- rule 6: obstacle veto ----------------

    @Test
    fun `obstacle veto when moving toward obstacle`() {
        val veto = DepthGrid(0.9f, floatArrayOf(1f, 0f)) // obstacle due north
        val d = gate.filter(obs(lastAction = floatArrayOf(0.5f, 0.1f, 0f)), null, 100f, 0f, veto)
        assertEquals("obstacle_veto", d.cause)
        assertAction(floatArrayOf(0f, 0f, 0f), d.action)
    }

    @Test
    fun `no veto when moving away from obstacle`() {
        val veto = DepthGrid(0.5f, floatArrayOf(1f, 0f)) // obstacle north
        val d = gate.filter(obs(lastAction = floatArrayOf(-0.5f, 0f, 0f)), null, 100f, 0f, veto)
        assertTrue(d.cause.startsWith("pass"), d.cause)
    }

    @Test
    fun `no veto when obstacle beyond depth veto distance`() {
        val veto = DepthGrid(1.5f, floatArrayOf(1f, 0f)) // 1.5 >= depthVetoMeters 1.0
        val d = gate.filter(obs(lastAction = floatArrayOf(0.5f, 0f, 0f)), null, 100f, 0f, veto)
        assertTrue(d.cause.startsWith("pass"), d.cause)
    }

    @Test
    fun `depth boundary equality does not veto - strictly less triggers`() {
        val exact = DepthGrid(1.0f, floatArrayOf(1f, 0f))
        val d1 = gate.filter(obs(lastAction = floatArrayOf(1f, 0f, 0f)), null, 100f, 0f, exact)
        assertTrue(d1.cause.startsWith("pass"), d1.cause)
        val just = DepthGrid(0.999f, floatArrayOf(1f, 0f))
        val d2 = gate.filter(obs(lastAction = floatArrayOf(1f, 0f, 0f)), null, 100f, 0f, just)
        assertEquals("obstacle_veto", d2.cause)
    }

    @Test
    fun `veto ignores vertical component of motion`() {
        // unit is (x,y,0): pure down move has dot 0 (not > 0) -> no veto
        val veto = DepthGrid(0.5f, floatArrayOf(1f, 0f))
        val d = gate.filter(obs(lastAction = floatArrayOf(0f, 0f, 0.8f)), null, 100f, 0f, veto)
        assertTrue(d.cause.startsWith("pass"), d.cause)
    }

    @Test
    fun `null depth grid never vetoes`() {
        val d = gate.filter(obs(lastAction = floatArrayOf(1f, 0f, 0f)), null, 100f, 0f, null)
        assertTrue(d.cause.startsWith("pass"), d.cause)
    }

    // ---------------- rule 7: pass, rate limit, speed cap ----------------

    @Test
    fun `passivity - unchanged action when in box with small deltas`() {
        val prev = floatArrayOf(0.1f, -0.2f, 0.05f)
        val policy = floatArrayOf(0.2f, -0.1f, 0.0f)
        val d = gate.filter(obs(posErr = floatArrayOf(0.5f, -0.4f, 0.2f)), prev, 100f, 0f, null, policy)
        assertEquals("pass", d.cause)
        assertAction(policy, d.action)
    }

    @Test
    fun `passivity with null prev action`() {
        val policy = floatArrayOf(0.3f, 0f, -0.2f)
        val d = gate.filter(obs(posErr = floatArrayOf(0f, 0f, 0.1f)), null, 100f, 0f, null, policy)
        assertEquals("pass", d.cause)
        assertAction(policy, d.action)
    }

    @Test
    fun `rate limit math prev zero policy one gives half`() {
        val d = gate.filter(obs(), null, 100f, 0f, null, floatArrayOf(1f, 1f, 1f))
        // prevAction non-null path is what matters for rate limiting:
        val d2 = gate.filter(obs(), floatArrayOf(0f, 0f, 0f), 100f, 0f, null, floatArrayOf(1f, 1f, 1f))
        assertEquals("pass|rate_limited", d2.cause)
        assertAction(floatArrayOf(0.5f, 0.5f, 0.5f), d2.action)
        // sanity: with prevAction null nothing to rate-limit
        assertEquals("pass", d.cause)
    }

    @Test
    fun `rate limit on lower bound`() {
        val d = gate.filter(obs(), floatArrayOf(0f, 0f, 0f), 100f, 0f, null, floatArrayOf(-1f, 0f, 0f))
        assertEquals("pass|rate_limited", d.cause)
        assertAction(floatArrayOf(-0.5f, 0f, 0f), d.action)
    }

    @Test
    fun `exact max delta change is not flagged rate limited`() {
        // prev=0, policy=maxDelta=0.4: allowed exactly, no flag
        val d = gate.filter(obs(), floatArrayOf(0f, 0f, 0f), 100f, 0f, null, floatArrayOf(0.4f, 0f, 0f))
        assertEquals("pass", d.cause)
        assertAction(floatArrayOf(0.4f, 0f, 0f), d.action)
    }

    @Test
    fun `speed cap clamps beyond maxSpeed`() {
        val d = gate.filter(obs(), null, 100f, 0f, null, floatArrayOf(2f, 0f, -1.5f))
        assertEquals("pass|speed_capped", d.cause)
        assertAction(floatArrayOf(1f, 0f, -1f), d.action)
    }

    @Test
    fun `both rate limited and speed capped`() {
        // prev=0.9, policy=1.6 -> rate limit to 1.4 (rate_limited) -> clamp to 1.0 (speed_capped)
        val d = gate.filter(obs(), floatArrayOf(0.9f, 0f, 0f), 100f, 0f, null, floatArrayOf(1.6f, 0f, 0f))
        assertEquals("pass|rate_limited|speed_capped", d.cause)
        assertAction(floatArrayOf(1f, 0f, 0f), d.action)
    }

    @Test
    fun `policy defaults to obs last action slot`() {
        // 5-parameter shape: policy command taken from obs[12..14]
        val d = gate.filter(obs(lastAction = floatArrayOf(0.7f, 0f, 0f)), null, 100f, 0f, null)
        assertEquals("pass", d.cause)
        assertAction(floatArrayOf(0.7f, 0f, 0f), d.action)
    }

    // ---------------- reset / state ----------------

    @Test
    fun `reset clears returnNeeded`() {
        val g = SafetyGate()
        g.filter(obs(), null, 20f, 0f, null)
        assertTrue(g.returnNeeded)
        g.reset()
        assertFalse(g.returnNeeded)
    }

    @Test
    fun `gate is stateless across calls - prevAction not cached`() {
        val g = SafetyGate()
        g.filter(obs(), floatArrayOf(0f, 0f, 0f), 100f, 0f, null, floatArrayOf(1f, 0f, 0f))
        // second call with prevAction=null must NOT rate-limit against the first call
        val d = g.filter(obs(), null, 100f, 0f, null, floatArrayOf(1f, 0f, 0f))
        assertEquals("pass", d.cause)
        assertAction(floatArrayOf(1f, 0f, 0f), d.action)
    }

    // ---------------- misc ----------------

    @Test
    fun `custom config overrides respected`() {
        val g = SafetyGate(SafetyConfig(hoverBoxRadius = 2f, geofenceRadius = 3f, maxSpeed = 0.5f, maxDeltaPerStep = 0.1f))
        val d = g.filter(obs(posErr = floatArrayOf(2f, 0f, 0f)), null, 100f, 0f, null)
        assertEquals("hover_box_pullin", d.cause)
        assertAction(floatArrayOf(-0.5f, 0f, 0f), d.action)
        val d2 = g.filter(obs(posErr = floatArrayOf(3f, 0f, 0f)), null, 100f, 0f, null)
        assertEquals("geofence_pullin", d2.cause)
    }

    @Test
    fun `decision data class exposes action and cause`() {
        val d = Decision(floatArrayOf(1f, 2f, 3f), "x")
        assertIterableEquals(listOf(1f, 2f, 3f), d.action.toList())
        assertEquals("x", d.cause)
    }
}
