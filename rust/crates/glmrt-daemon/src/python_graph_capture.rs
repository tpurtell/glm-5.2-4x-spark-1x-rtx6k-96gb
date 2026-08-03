use anyhow::{Context, Result};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};
#[cfg(test)]
use std::cell::Cell;
use std::env;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

pub(crate) const GLMRT_B12X_ENV: &str = "GLMRT_B12X";
pub(crate) const GLMRT_B12X_SPARK_ENV: &str = "GLMRT_B12X_SPARK_PYTHON_CAPTURE";
pub(crate) const GLMRT_SPARK_LAYER_BLOCK_ATTENTION_CAPTURE_ENV: &str =
    "GLMRT_SPARK_LAYER_BLOCK_ATTENTION_PYTHON_CAPTURE";

const COORDINATOR_PYTHON_CAPTURE_MODULES: &[&str] = &[
    "sparkinfer",
    "flashinfer",
    "triton",
    "b12x_mla_capture",
    "dspark_capture",
    "triton_mlp_capture",
    "triton_router_capture",
    "triton_sampling_capture",
    "triton_kv_pack_capture",
];
const SPARK_PYTHON_CAPTURE_MODULES: &[&str] = &["b12x_spark_capture"];
const SPARK_LAYER_BLOCK_ATTENTION_CAPTURE_MODULES: &[&str] =
    &["sparkinfer", "flashinfer", "b12x_mla_capture"];
static COORDINATOR_PYTHON_CAPTURE_STARTUP_OPEN: AtomicBool = AtomicBool::new(true);

#[cfg(test)]
thread_local! {
    static COORDINATOR_PYTHON_CAPTURE_TEST_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PythonCaptureStatus {
    pub(crate) gate_env: &'static str,
    pub(crate) imported_modules: Vec<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct PythonDeviceBufferArg<'a> {
    pub(crate) name: &'a str,
    pub(crate) ptr: *mut c_void,
    pub(crate) bytes: usize,
    pub(crate) device_id: i32,
    pub(crate) flags: u64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) enum PythonKernelArg<'a> {
    Bool(bool),
    F64(f64),
    I64(i64),
    Str(&'a str),
    Usize(usize),
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct PythonGraphCaptureLaunch<'a> {
    pub(crate) module: &'a str,
    pub(crate) function: &'a str,
    pub(crate) cuda_stream: *mut c_void,
    pub(crate) buffers: &'a [PythonDeviceBufferArg<'a>],
    pub(crate) kwargs: &'a [(&'a str, PythonKernelArg<'a>)],
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PythonBoolQuery<'a> {
    pub(crate) module: &'a str,
    pub(crate) function: &'a str,
    pub(crate) kwargs: &'a [(&'a str, PythonKernelArg<'a>)],
}

pub(crate) fn initialize_coordinator_python_capture_from_env() -> Result<Option<PythonCaptureStatus>>
{
    if !coordinator_python_capture_enabled() {
        return Ok(None);
    }

    initialize_python_capture(
        GLMRT_B12X_ENV,
        COORDINATOR_PYTHON_CAPTURE_MODULES,
        "coordinator",
    )
}

pub(crate) fn initialize_spark_python_capture_from_env() -> Result<Option<PythonCaptureStatus>> {
    if !spark_python_capture_enabled() {
        return Ok(None);
    }

    initialize_python_capture(GLMRT_B12X_SPARK_ENV, SPARK_PYTHON_CAPTURE_MODULES, "Spark")
}

pub(crate) fn initialize_spark_layer_block_attention_capture_from_env(
) -> Result<Option<PythonCaptureStatus>> {
    if !spark_layer_block_attention_capture_enabled() {
        return Ok(None);
    }

    initialize_python_capture(
        GLMRT_SPARK_LAYER_BLOCK_ATTENTION_CAPTURE_ENV,
        SPARK_LAYER_BLOCK_ATTENTION_CAPTURE_MODULES,
        "Spark layer-block attention",
    )
}

fn initialize_python_capture(
    gate_env: &'static str,
    modules: &[&str],
    label: &str,
) -> Result<Option<PythonCaptureStatus>> {
    let startup_started = Instant::now();
    let prepare_started = Instant::now();
    pyo3::prepare_freethreaded_python();
    eprintln!(
        "python_capture_startup_phase component={label:?} stage=python-runtime elapsed_ms={:.3} total_ms={:.3}",
        prepare_started.elapsed().as_secs_f64() * 1_000.0,
        startup_started.elapsed().as_secs_f64() * 1_000.0,
    );
    Python::with_gil(|py| -> PyResult<Vec<String>> {
        let path_started = Instant::now();
        add_glmrt_python_reference_path(py)?;
        eprintln!(
            "python_capture_startup_phase component={label:?} stage=python-path elapsed_ms={:.3} total_ms={:.3}",
            path_started.elapsed().as_secs_f64() * 1_000.0,
            startup_started.elapsed().as_secs_f64() * 1_000.0,
        );
        let mut imported_modules = Vec::with_capacity(modules.len());
        for module in modules {
            let import_started = Instant::now();
            PyModule::import_bound(py, *module)?;
            eprintln!(
                "python_capture_startup_phase component={label:?} stage=module-import module={module} elapsed_ms={:.3} total_ms={:.3}",
                import_started.elapsed().as_secs_f64() * 1_000.0,
                startup_started.elapsed().as_secs_f64() * 1_000.0,
            );
            imported_modules.push((*module).to_owned());
        }
        Ok(imported_modules)
    })
    .map(|imported_modules| {
        Some(PythonCaptureStatus {
            gate_env,
            imported_modules,
        })
    })
    .map_err(|err| anyhow::anyhow!(format_python_error(err)))
    .with_context(|| format!("importing {label} Python kernel modules"))
    .map(|status| {
        eprintln!(
            "python_capture_startup_phase component={label:?} stage=complete elapsed_ms={:.3} total_ms={:.3}",
            startup_started.elapsed().as_secs_f64() * 1_000.0,
            startup_started.elapsed().as_secs_f64() * 1_000.0,
        );
        status
    })
}

pub(crate) fn coordinator_python_capture_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = COORDINATOR_PYTHON_CAPTURE_TEST_OVERRIDE.with(|value| value.get()) {
        return enabled;
    }

    env::var(GLMRT_B12X_ENV)
        .map(|value| matches_env_true(&value))
        .unwrap_or(false)
}

pub(crate) fn attention_python_capture_enabled() -> bool {
    coordinator_python_capture_enabled() || spark_layer_block_attention_capture_enabled()
}

pub(crate) fn finish_coordinator_python_capture_startup() {
    COORDINATOR_PYTHON_CAPTURE_STARTUP_OPEN.store(false, Ordering::Release);
}

pub(crate) fn coordinator_python_capture_startup_open() -> bool {
    COORDINATOR_PYTHON_CAPTURE_STARTUP_OPEN.load(Ordering::Acquire)
}

pub(crate) fn spark_python_capture_enabled() -> bool {
    env::var(GLMRT_B12X_SPARK_ENV)
        .map(|value| matches_env_true(&value))
        .unwrap_or(false)
}

pub(crate) fn spark_layer_block_attention_capture_enabled() -> bool {
    env::var(GLMRT_SPARK_LAYER_BLOCK_ATTENTION_CAPTURE_ENV)
        .map(|value| matches_env_true(&value))
        .unwrap_or(false)
}

#[allow(dead_code)]
pub(crate) fn launch_python_graph_capture(launch: PythonGraphCaptureLaunch<'_>) -> Result<()> {
    anyhow::ensure!(
        coordinator_python_capture_startup_open() || launch.module == "b12x_spark_capture",
        "coordinator Python graph capture is closed after startup"
    );
    anyhow::ensure!(
        !launch.cuda_stream.is_null(),
        "Python graph-capture launch requires a non-null CUDA stream"
    );

    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| call_python_graph_capture(py, &launch).map(|_| ()))
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))
        .with_context(|| {
            format!(
                "launching Python graph-capture kernel {}.{}",
                launch.module, launch.function
            )
        })
}

pub(crate) fn launch_python_kernel(launch: PythonGraphCaptureLaunch<'_>) -> Result<()> {
    anyhow::ensure!(
        !launch.cuda_stream.is_null(),
        "Python kernel launch requires a non-null CUDA stream"
    );

    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| call_python_graph_capture(py, &launch).map(|_| ()))
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))
        .with_context(|| {
            format!(
                "launching Python kernel {}.{}",
                launch.module, launch.function
            )
        })
}

pub(crate) fn query_python_bool_during_startup(query: PythonBoolQuery<'_>) -> Result<bool> {
    anyhow::ensure!(
        coordinator_python_capture_startup_open(),
        "Python planner queries are closed after coordinator startup"
    );

    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| call_python_bool_query(py, &query))
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))
        .with_context(|| {
            format!(
                "querying Python planner {}.{}",
                query.module, query.function
            )
        })
}

fn call_python_graph_capture<'py>(
    py: Python<'py>,
    launch: &PythonGraphCaptureLaunch<'_>,
) -> PyResult<pyo3::Bound<'py, PyAny>> {
    add_glmrt_python_reference_path(py)?;
    let module = PyModule::import_bound(py, launch.module)?;
    let function = module.getattr(launch.function)?;
    let context = PyDict::new_bound(py);
    context.set_item("cuda_stream", launch.cuda_stream as usize)?;
    context.set_item("cuda_stream_ptr", launch.cuda_stream as usize)?;
    context.set_item("capture_phase", "cuda_graph_capture")?;

    let buffers = PyDict::new_bound(py);
    for buffer in launch.buffers {
        let py_buffer = PyDict::new_bound(py);
        py_buffer.set_item("ptr", buffer.ptr as usize)?;
        py_buffer.set_item("bytes", buffer.bytes)?;
        py_buffer.set_item("device_id", buffer.device_id)?;
        py_buffer.set_item("flags", buffer.flags)?;
        buffers.set_item(buffer.name, py_buffer)?;
    }
    context.set_item("buffers", buffers)?;

    let kwargs = PyDict::new_bound(py);
    for (name, value) in launch.kwargs {
        match value {
            PythonKernelArg::Bool(value) => kwargs.set_item(name, value)?,
            PythonKernelArg::F64(value) => kwargs.set_item(name, value)?,
            PythonKernelArg::I64(value) => kwargs.set_item(name, value)?,
            PythonKernelArg::Str(value) => kwargs.set_item(name, value)?,
            PythonKernelArg::Usize(value) => kwargs.set_item(name, value)?,
        }
    }

    function.call((context,), Some(&kwargs))
}

fn call_python_bool_query(py: Python<'_>, query: &PythonBoolQuery<'_>) -> PyResult<bool> {
    add_glmrt_python_reference_path(py)?;
    let module = PyModule::import_bound(py, query.module)?;
    let function = module.getattr(query.function)?;
    let kwargs = PyDict::new_bound(py);
    for (name, value) in query.kwargs {
        match value {
            PythonKernelArg::Bool(value) => kwargs.set_item(name, value)?,
            PythonKernelArg::F64(value) => kwargs.set_item(name, value)?,
            PythonKernelArg::I64(value) => kwargs.set_item(name, value)?,
            PythonKernelArg::Str(value) => kwargs.set_item(name, value)?,
            PythonKernelArg::Usize(value) => kwargs.set_item(name, value)?,
        }
    }
    function.call((), Some(&kwargs))?.extract::<bool>()
}

fn add_glmrt_python_reference_path(py: Python<'_>) -> PyResult<()> {
    let sys = PyModule::import_bound(py, "sys")?;
    let sys_path = sys.getattr("path")?;
    for path in glmrt_python_reference_paths() {
        add_python_path(&sys_path, path)?;
    }
    for path in glmrt_python_dynload_paths(py)? {
        add_python_path(&sys_path, path)?;
    }
    Ok(())
}

fn add_python_path(sys_path: &pyo3::Bound<'_, PyAny>, path: PathBuf) -> PyResult<()> {
    if !path.is_dir() {
        return Ok(());
    }
    let path = path.to_string_lossy().to_string();
    if !sys_path.contains(path.as_str())? {
        sys_path.call_method1("insert", (0, path))?;
    }
    Ok(())
}

fn glmrt_python_reference_paths() -> [PathBuf; 2] {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join("python")
        .join("reference");
    [root.clone(), root.join("glmrt_reference")]
}

fn glmrt_python_dynload_paths(py: Python<'_>) -> PyResult<Vec<PathBuf>> {
    let sys = PyModule::import_bound(py, "sys")?;
    let version_info = sys.getattr("version_info")?;
    let major = version_info.get_item(0)?.extract::<usize>()?;
    let minor = version_info.get_item(1)?.extract::<usize>()?;
    let python_version = format!("python{major}.{minor}");

    let mut prefixes = Vec::new();
    if let Ok(prefix) = sys.getattr("prefix")?.extract::<String>() {
        prefixes.push(PathBuf::from(prefix));
    }
    if let Ok(prefix) = sys.getattr("base_prefix")?.extract::<String>() {
        prefixes.push(PathBuf::from(prefix));
    }
    if let Some(prefix) = env::var_os("PYTHONHOME") {
        prefixes.push(PathBuf::from(prefix));
    }
    if let Some(prefix) = env::var_os("VIRTUAL_ENV") {
        prefixes.push(PathBuf::from(prefix));
    }
    prefixes.push(PathBuf::from("/usr"));
    prefixes.push(PathBuf::from("/usr/local"));

    let mut paths = Vec::new();
    for prefix in prefixes {
        for lib_dir in ["lib", "lib64"] {
            let path = prefix
                .join(lib_dir)
                .join(&python_version)
                .join("lib-dynload");
            if !paths.contains(&path) {
                paths.push(path);
            }
        }
    }
    Ok(paths)
}

fn matches_env_true(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on" | "enabled"
    )
}

fn format_python_error(err: PyErr) -> String {
    err.to_string()
}

#[cfg(test)]
pub(crate) struct CoordinatorPythonCaptureTestOverride {
    previous: Option<bool>,
}

#[cfg(test)]
impl Drop for CoordinatorPythonCaptureTestOverride {
    fn drop(&mut self) {
        set_coordinator_python_capture_test_override(self.previous);
    }
}

#[cfg(test)]
pub(crate) fn set_coordinator_python_capture_test_override(enabled: Option<bool>) -> Option<bool> {
    COORDINATOR_PYTHON_CAPTURE_TEST_OVERRIDE.with(|value| {
        let previous = value.get();
        value.set(enabled);
        previous
    })
}

#[cfg(test)]
pub(crate) fn coordinator_python_capture_test_override(
    enabled: bool,
) -> CoordinatorPythonCaptureTestOverride {
    CoordinatorPythonCaptureTestOverride {
        previous: set_coordinator_python_capture_test_override(Some(enabled)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn env_gate_defaults_to_disabled() {
        let _guard = ENV_MUTEX.lock().expect("env test mutex poisoned");
        env::remove_var(GLMRT_B12X_ENV);
        env::remove_var(GLMRT_B12X_SPARK_ENV);
        assert!(!coordinator_python_capture_enabled());
        assert!(!spark_python_capture_enabled());
    }

    #[test]
    fn env_gates_parse_true_values() {
        let _guard = ENV_MUTEX.lock().expect("env test mutex poisoned");
        env::set_var(GLMRT_B12X_ENV, "on");
        env::set_var(GLMRT_B12X_SPARK_ENV, "1");
        assert!(coordinator_python_capture_enabled());
        assert!(spark_python_capture_enabled());
        env::remove_var(GLMRT_B12X_ENV);
        env::remove_var(GLMRT_B12X_SPARK_ENV);
    }

    #[test]
    fn legacy_spark_b12x_env_does_not_enable_python_capture() {
        let _guard = ENV_MUTEX.lock().expect("env test mutex poisoned");
        env::remove_var(GLMRT_B12X_SPARK_ENV);
        env::set_var("GLMRT_B12X_SPARK", "1");
        assert!(!spark_python_capture_enabled());
        env::remove_var("GLMRT_B12X_SPARK");
    }

    #[test]
    fn launch_passes_stream_buffers_and_kwargs_to_python() -> Result<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            let module = PyModule::from_code_bound(
                py,
                r#"
captured = None
def capture(ctx, *, rows, label, deterministic):
    global captured
    captured = {
        "stream": ctx["cuda_stream"],
        "phase": ctx["capture_phase"],
        "x_ptr": ctx["buffers"]["x"]["ptr"],
        "x_bytes": ctx["buffers"]["x"]["bytes"],
        "device_id": ctx["buffers"]["x"]["device_id"],
        "rows": rows,
        "label": label,
        "deterministic": deterministic,
    }
"#,
                "glmrt_test_capture.py",
                "glmrt_test_capture",
            )?;
            let sys = PyModule::import_bound(py, "sys")?;
            sys.getattr("modules")?
                .set_item("glmrt_test_capture", module)?;
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))?;

        let stream = 0x1234usize as *mut c_void;
        let ptr = 0x5678usize as *mut c_void;
        let buffers = [PythonDeviceBufferArg {
            name: "x",
            ptr,
            bytes: 4096,
            device_id: 0,
            flags: 7,
        }];
        let kwargs = [
            ("rows", PythonKernelArg::Usize(16)),
            ("label", PythonKernelArg::Str("unit-test")),
            ("deterministic", PythonKernelArg::Bool(true)),
        ];
        launch_python_graph_capture(PythonGraphCaptureLaunch {
            module: "glmrt_test_capture",
            function: "capture",
            cuda_stream: stream,
            buffers: &buffers,
            kwargs: &kwargs,
        })?;

        Python::with_gil(|py| -> PyResult<()> {
            let module = PyModule::import_bound(py, "glmrt_test_capture")?;
            let captured = module.getattr("captured")?;
            assert_eq!(
                captured.get_item("stream")?.extract::<usize>()?,
                stream as usize
            );
            assert_eq!(
                captured.get_item("phase")?.extract::<String>()?,
                "cuda_graph_capture"
            );
            assert_eq!(
                captured.get_item("x_ptr")?.extract::<usize>()?,
                ptr as usize
            );
            assert_eq!(captured.get_item("x_bytes")?.extract::<usize>()?, 4096);
            assert_eq!(captured.get_item("device_id")?.extract::<i32>()?, 0);
            assert_eq!(captured.get_item("rows")?.extract::<usize>()?, 16);
            assert_eq!(
                captured.get_item("label")?.extract::<String>()?,
                "unit-test"
            );
            assert!(captured.get_item("deterministic")?.extract::<bool>()?);
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))?;

        Ok(())
    }

    #[test]
    fn startup_bool_query_passes_kwargs_and_extracts_result() -> Result<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            let module = PyModule::from_code_bound(
                py,
                r#"
def plan(*, workload, rows, enabled):
    return workload == "decode" and rows == 8 and enabled
"#,
                "glmrt_test_bool_query.py",
                "glmrt_test_bool_query",
            )?;
            let sys = PyModule::import_bound(py, "sys")?;
            sys.getattr("modules")?
                .set_item("glmrt_test_bool_query", module)?;
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))?;

        let kwargs = [
            ("workload", PythonKernelArg::Str("decode")),
            ("rows", PythonKernelArg::Usize(8)),
            ("enabled", PythonKernelArg::Bool(true)),
        ];
        assert!(query_python_bool_during_startup(PythonBoolQuery {
            module: "glmrt_test_bool_query",
            function: "plan",
            kwargs: &kwargs,
        })?);

        Ok(())
    }

    #[test]
    fn launch_imports_glmrt_reference_b12x_adapter_with_target_override() -> Result<()> {
        let _guard = ENV_MUTEX.lock().expect("env test mutex poisoned");
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            let module = PyModule::from_code_bound(
                py,
                r#"
captured = None
def capture(ctx, **kwargs):
    global captured
    captured = {
        "phase": ctx["capture_phase"],
        "rows": kwargs["rows"],
    }
"#,
                "glmrt_test_b12x_capture_target.py",
                "glmrt_test_b12x_capture_target",
            )?;
            let sys = PyModule::import_bound(py, "sys")?;
            sys.getattr("modules")?
                .set_item("glmrt_test_b12x_capture_target", module)?;
            let os = PyModule::import_bound(py, "os")?;
            os.getattr("environ")?.set_item(
                "GLMRT_B12X_MLA_CAPTURE_TARGET",
                "glmrt_test_b12x_capture_target:capture",
            )?;
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))?;

        env::set_var(
            "GLMRT_B12X_MLA_CAPTURE_TARGET",
            "glmrt_test_b12x_capture_target:capture",
        );
        let kwargs = [("rows", PythonKernelArg::Usize(4))];
        let result = launch_python_graph_capture(PythonGraphCaptureLaunch {
            module: "b12x_mla_capture",
            function: "capture_mla_rope_attention",
            cuda_stream: 0x1234usize as *mut c_void,
            buffers: &[],
            kwargs: &kwargs,
        });
        env::remove_var("GLMRT_B12X_MLA_CAPTURE_TARGET");
        Python::with_gil(|py| -> PyResult<()> {
            let os = PyModule::import_bound(py, "os")?;
            os.getattr("environ")?
                .call_method1("pop", ("GLMRT_B12X_MLA_CAPTURE_TARGET", py.None()))?;
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))?;
        result?;

        Python::with_gil(|py| -> PyResult<()> {
            let module = PyModule::import_bound(py, "glmrt_test_b12x_capture_target")?;
            let captured = module.getattr("captured")?;
            assert_eq!(
                captured.get_item("phase")?.extract::<String>()?,
                "cuda_graph_capture"
            );
            assert_eq!(captured.get_item("rows")?.extract::<usize>()?, 4);
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))?;

        Ok(())
    }

    #[test]
    fn launch_imports_glmrt_reference_b12x_spark_adapter_with_target_override() -> Result<()> {
        let _guard = ENV_MUTEX.lock().expect("env test mutex poisoned");
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            let module = PyModule::from_code_bound(
                py,
                r#"
captured = None
def capture(ctx, **kwargs):
    global captured
    captured = {
        "phase": ctx["capture_phase"],
        "rows": kwargs["rows"],
        "n": kwargs["n"],
        "k": kwargs["k"],
    }
"#,
                "glmrt_test_b12x_spark_capture_target.py",
                "glmrt_test_b12x_spark_capture_target",
            )?;
            let sys = PyModule::import_bound(py, "sys")?;
            sys.getattr("modules")?
                .set_item("glmrt_test_b12x_spark_capture_target", module)?;
            let os = PyModule::import_bound(py, "os")?;
            os.getattr("environ")?.set_item(
                "GLMRT_B12X_SPARK_CAPTURE_TARGET",
                "glmrt_test_b12x_spark_capture_target:capture",
            )?;
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))?;

        env::set_var(
            "GLMRT_B12X_SPARK_CAPTURE_TARGET",
            "glmrt_test_b12x_spark_capture_target:capture",
        );
        let kwargs = [
            ("rows", PythonKernelArg::Usize(4)),
            ("n", PythonKernelArg::Usize(16)),
            ("k", PythonKernelArg::Usize(32)),
        ];
        let result = launch_python_graph_capture(PythonGraphCaptureLaunch {
            module: "b12x_spark_capture",
            function: "capture_dense_gemm",
            cuda_stream: 0x1234usize as *mut c_void,
            buffers: &[],
            kwargs: &kwargs,
        });
        env::remove_var("GLMRT_B12X_SPARK_CAPTURE_TARGET");
        Python::with_gil(|py| -> PyResult<()> {
            let os = PyModule::import_bound(py, "os")?;
            os.getattr("environ")?
                .call_method1("pop", ("GLMRT_B12X_SPARK_CAPTURE_TARGET", py.None()))?;
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))?;
        result?;

        Python::with_gil(|py| -> PyResult<()> {
            let module = PyModule::import_bound(py, "glmrt_test_b12x_spark_capture_target")?;
            let captured = module.getattr("captured")?;
            assert_eq!(
                captured.get_item("phase")?.extract::<String>()?,
                "cuda_graph_capture"
            );
            assert_eq!(captured.get_item("rows")?.extract::<usize>()?, 4);
            assert_eq!(captured.get_item("n")?.extract::<usize>()?, 16);
            assert_eq!(captured.get_item("k")?.extract::<usize>()?, 32);
            Ok(())
        })
        .map_err(|err| anyhow::anyhow!(format_python_error(err)))?;

        Ok(())
    }
}
