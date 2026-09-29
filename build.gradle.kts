// Root project: IDE setup and helper tasks that run the Python core via uv.
// The Java GUI lives in the :gui subproject.

plugins {
    idea
}

idea {
    module {
        // Keep IntelliJ from indexing the Python venv (several GB), data and outputs.
        excludeDirs.addAll(
            listOf(
                "core/.venv",
                "core/.pytest_cache",
                "data",
                "experiments/runs",
                "build",
                "gui/build",
            ).map { file(it) }
        )
    }
}

val coreDir = layout.projectDirectory.dir("core")

tasks.register<Exec>("coreSync") {
    group = "python core"
    description = "Installs Python core dependencies, including PyTorch with CUDA (uv sync --extra nn)."
    workingDir(coreDir)
    commandLine("uv", "sync", "--extra", "nn")
}

tasks.register<Exec>("coreTest") {
    group = "python core"
    description = "Runs the Python core tests (uv run pytest)."
    workingDir(coreDir)
    commandLine("uv", "run", "pytest")
}

tasks.register<Exec>("coreInfo") {
    group = "python core"
    description = "Shows Python core package versions and CUDA status."
    workingDir(coreDir)
    commandLine("uv", "run", "spacemidi", "info")
}
