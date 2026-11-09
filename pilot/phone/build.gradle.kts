plugins {
    kotlin("jvm") version "1.9.22"
    application
}

repositories {
    mavenCentral()
    google()
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime:1.19.2")
    implementation("com.fasterxml.jackson.module:jackson-module-kotlin:2.17.2")

    testImplementation(platform("org.junit:junit-bom:5.10.2"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

kotlin {
    jvmToolchain(17)
}

application {
    mainClass.set("miniai.pilot.LoopbackRunnerKt")
}

tasks.test {
    useJUnitPlatform()
    testLogging {
        events("passed", "failed", "skipped")
    }
}
