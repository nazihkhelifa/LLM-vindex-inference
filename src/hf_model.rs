//! Detect Hugging Face model folders (config + safetensors) and optionally download via the `hf` CLI.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Resolve a user-supplied path against `std::env::current_dir()` when it is relative (recommended on Windows for `hf download`).
pub fn absolutize_model_dir(model_dir: &Path) -> io::Result<PathBuf> {
    if model_dir.is_absolute() {
        Ok(model_dir.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(model_dir))
    }
}

/// Default local folder when `--hf-repo` is set but `--model-dir` is omitted: `./hf_checkout/<repo with "/" → "__">`.
pub fn default_checkout_dir_for_repo(repo: &str) -> io::Result<PathBuf> {
    let slug = repo.trim().replace(['/', '\\'], "__");
    Ok(std::env::current_dir()?.join("hf_checkout").join(slug))
}

/// Resolve `--model-dir` to an absolute path, or if omitted and `hf_repo` is set, use [`default_checkout_dir_for_repo`].
pub fn resolve_model_dir_for_cli(
    model_dir: Option<&Path>,
    hf_repo: Option<&str>,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    match (model_dir, hf_repo) {
        (Some(p), _) => Ok(absolutize_model_dir(p)?),
        (None, Some(repo)) => Ok(default_checkout_dir_for_repo(repo)?),
        (None, None) => Err(
            "Missing --model-dir. Pass --model-dir PATH to your HF snapshot, or pass --hf-repo alone to download under .\\hf_checkout\\<repo>/"
                .into(),
        ),
    }
}

/// True if `config.json` exists and at least one `*.safetensors` file is in `dir` (non-recursive).
pub fn hf_model_has_safetensors(dir: &Path) -> bool {
    let config = dir.join("config.json");
    if !config.is_file() {
        return false;
    }
    let Ok(rd) = fs::read_dir(dir) else {
        return false;
    };
    rd.flatten().any(|e| {
        e.path()
            .extension()
            .and_then(|s| s.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("safetensors"))
            .unwrap_or(false)
    })
}

pub fn count_safetensors(dir: &Path) -> usize {
    let Ok(rd) = fs::read_dir(dir) else {
        return 0;
    };
    rd.flatten()
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|s| s.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("safetensors"))
                .unwrap_or(false)
        })
        .count()
}

pub fn diagnose(model_dir: &Path) -> String {
    let config = model_dir.join("config.json");
    let has_config = config.is_file();
    let n = count_safetensors(model_dir);
    format!(
        "model_dir={}\n  config.json: {}\n  *.safetensors in directory: {}",
        model_dir.display(),
        if has_config { "present" } else { "MISSING" },
        n
    )
}

/// Run `hf download <repo> --local-dir <model_dir>`. Requires [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (`pip install huggingface_hub` then `hf` on PATH).
///
/// `hf_token`: forwarded to the child as `HF_TOKEN` when non-empty (gated models such as Gemma).
pub fn hf_download_via_cli(
    repo: &str,
    local_dir: &Path,
    hf_token: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let local_str = local_dir
        .to_str()
        .ok_or("model-dir must be valid UTF-8 for the hf CLI")?;

    let token_set = hf_token.map(|t| !t.trim().is_empty()).unwrap_or(false);
    log::info!(
        "hf download step 1/3: spawn `hf download {} --local-dir {}` (HF_TOKEN set: {})",
        repo,
        local_str,
        token_set
    );
    log::info!(
        "hf download step 2/3: streaming hf stdout/stderr (tqdm progress appears here)…"
    );

    let mut cmd = Command::new("hf");
    cmd.args(["download", repo, "--local-dir", local_str]);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());
    if let Some(t) = hf_token.map(|t| t.trim()).filter(|t| !t.is_empty()) {
        cmd.env("HF_TOKEN", t);
    }

    let status = cmd.status().map_err(|e| {
        if e.kind() == io::ErrorKind::NotFound {
            io::Error::new(
                io::ErrorKind::NotFound,
                "`hf` not found on PATH. Install with: pip install huggingface_hub",
            )
        } else {
            e
        }
    })?;

    if !status.success() {
        return Err(format!(
            "`hf download` failed (exit {:?}). Read the `hf` output above for the real cause.\n\n\
             Common causes:\n\
             • No space left on device (Python OSError errno 28): Gemma 3 4B needs roughly 9–10+ GB free on the drive that contains:\n\
               {}\n\
               Free space, empty the recycle bin, remove old models, or use --model-dir on another disk.\n\
             • 401 / gated repo: accept the license on the model page at huggingface.co, then hf login, or HF_TOKEN, or --hf-token.\n\
             • Partial or corrupt download: delete the target folder and retry.\n",
            status.code(),
            local_str
        )
        .into());
    }

    log::info!("hf download step 3/3: subprocess exited successfully");
    Ok(())
}

/// If the folder already has `config.json` and at least one `.safetensors`, no-op.
/// Otherwise downloads with `hf download` when `hf_repo` is set.
pub fn ensure_hf_weights(
    model_dir: &Path,
    hf_repo: Option<&str>,
    hf_token: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_abs = absolutize_model_dir(model_dir).map_err(|e| {
        format!(
            "Could not resolve model directory {} (cwd?): {}",
            model_dir.display(),
            e
        )
    })?;

    if hf_model_has_safetensors(&model_abs) {
        log::info!(
            "HF weights already present under {} ({} safetensor shard(s)); skipping download",
            model_abs.display(),
            count_safetensors(&model_abs)
        );
        return Ok(());
    }

    let Some(repo) = hf_repo else {
        return Err(format!(
            "HF model folder is incomplete (need config.json and at least one *.safetensors) under {}.\n\
             Pass --hf-repo org/model-id to download automatically with the `hf` CLI, or run:\n\
               hf download <repo> --local-dir {}",
            model_abs.display(),
            model_abs.display()
        )
        .into());
    };

    log::info!(
        "HF weights missing or incomplete under {}",
        model_abs.display()
    );
    log::info!(
        "HF prepare: will download repo `{}` into `{}`",
        repo,
        model_abs.display()
    );

    // Single create_dir_all (avoids Windows ERROR_PATH_NOT_FOUND from create_dir_all on a drive root parent).
    log::info!("HF prepare: create_dir_all `{}`", model_abs.display());
    fs::create_dir_all(&model_abs).map_err(|e| {
        format!(
            "Could not create model directory {}: {}",
            model_abs.display(),
            e
        )
    })?;

    log::info!(
        "HF download: target `{}` — ensure this drive has enough free space (Gemma 3 4B weights are on the order of 9–10 GB).",
        model_abs.display()
    );

    hf_download_via_cli(repo, &model_abs, hf_token)?;

    log::info!(
        "HF verify: config + shards — {}",
        diagnose(&model_abs).replace('\n', " | ")
    );
    if !hf_model_has_safetensors(&model_abs) {
        return Err(format!(
            "After download, expected files still missing under {}.\n{}",
            model_abs.display(),
            diagnose(&model_abs)
        )
        .into());
    }

    log::info!(
        "HF download pipeline complete: {} safetensor shard(s) under {}",
        count_safetensors(&model_abs),
        model_abs.display()
    );

    Ok(())
}
