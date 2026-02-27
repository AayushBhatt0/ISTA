"""
================================================================================
                        TPU Pod Orchestration Script
================================================================================

Automates the full lifecycle of training on a Google Cloud TPU v5e pod:

    1. Upload latest code to GCS bucket
    2. (Optional) Recycle the TPU pod entirely
    3. Trust SSH host keys on all 8 workers
    4. Install dependencies and launch training
    5. Tail logs from worker 0

Usage:
    python TPU.py

    The script prompts for zone selection and whether to reuse an existing pod
    or create a fresh one. After training finishes or crashes, press Enter to
    repeat the full cycle (useful for hyperparameter sweeps or crash recovery).

Hardware:
    TPU v5litepod-32 — 32 cores across 8 workers (4 cores/worker)
    Zones: europe-west4-b (default) or us-central1-a

================================================================================
"""

import subprocess
import sys

# ============================================================================
# Configuration
# Set these before running. TPU_NAME and bucket names are project-specific.
# ============================================================================

TPU_NAME = ""           # e.g. "ista-tpu"
DEFAULT_ZONE = ""       # e.g. "europe-west4-b"
ALT_ZONE = ""           # e.g. "us-central1-a"

DEFAULT_BUCKET = ""     # GCS bucket for DEFAULT_ZONE
US_BUCKET = ""          # GCS bucket for ALT_ZONE

# ============================================================================
# Helpers
# ============================================================================

def run_ps(script):
    """Run a multi-line string as a PowerShell script, raising on failure."""
    subprocess.run(["powershell", "-Command", script], check=True)


def run(cmd):
    """Run a shell command string, raising on non-zero exit code."""
    subprocess.run(cmd, shell=True, check=True)


def choose_zone():
    """Prompt for zone selection and return the chosen zone string."""
    choice = input(
        "Select zone: [e]urope-west4-b (default) or [u]s-central1-a? [e/u]: "
    ).strip().lower()
    return ALT_ZONE if choice.startswith("u") else DEFAULT_ZONE


def bucket_for_zone(zone):
    """Return the GCS bucket associated with the given zone."""
    return US_BUCKET if zone == ALT_ZONE else DEFAULT_BUCKET


def tail_logs(zone):
    """Print existing log then live-tail it from worker 0, then exit."""
    tail_script = f"""
    $TPU_NAME = "{TPU_NAME}"
    $ZONE = "{zone}"

    Write-Host "--- CAT LOGS (worker 0) ---" -ForegroundColor Cyan
    echo "" | gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0 --tunnel-through-iap --command="cat v5.log"

    Write-Host "--- TAILING LOGS (worker 0) ---" -ForegroundColor Green
    echo "" | gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0 --tunnel-through-iap --command="stdbuf -oL tail -f v5.log"
    """
    run_ps(tail_script)
    sys.exit(0)


# ============================================================================
# Main Loop
# Repeats after each training run so crashes can be recovered without
# restarting the script manually.
# ============================================================================

while True:
    try:
        ZONE = choose_zone()
        BUCKET = bucket_for_zone(ZONE)

        mode = input("Tail logs only (t) or restart TPU (r)? [t/r]: ").strip().lower()
        if mode.startswith("t"):
            tail_logs(ZONE)

        reuse = input("Reuse existing TPU? (y/N): ").strip().lower() == "y"

        # ====================================================================
        # Upload Code
        # Always upload first so workers get the latest main.py regardless
        # of whether the pod is fresh or reused.
        # ====================================================================
        print("--- UPLOADING CODE ---")
        run(f"gcloud storage cp main.py gs://{BUCKET}/main.py")

        # ====================================================================
        # Fresh TPU Pod Path
        # Delete the existing pod, create a new spot instance, then set up
        # all 8 workers from scratch before launching.
        # ====================================================================
        if not reuse:
            print("--- RECYCLING TPU ---")
            run(f"gcloud compute tpus tpu-vm delete {TPU_NAME} --zone={ZONE} --quiet")
            run(f"gcloud compute tpus tpu-vm create {TPU_NAME} --zone={ZONE} --accelerator-type=v5litepod-32 --version=v2-tpuv5-litepod --internal-ips --spot")

            # ----------------------------------------------------------------
            # Trust SSH host keys on all 8 workers.
            # Must be done interactively (once per fresh pod) before batch SSH
            # can proceed without interactive prompts.
            # ----------------------------------------------------------------
            print("--- TRUSTING WORKER KEYS ---")
            trust_script = f"""
            $TPU_NAME = "{TPU_NAME}"
            $ZONE = "{ZONE}"
            foreach ($i in 0..7) {{
                Write-Host "Trusting Worker $i..." -ForegroundColor Yellow
                echo "y" | gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=$i --tunnel-through-iap --command="exit"
            }}
            """
            run_ps(trust_script)

            # ----------------------------------------------------------------
            # Phase 1: Install deps and sync code on all workers.
            # Phase 2: Launch training in background — nohup keeps it alive
            #          after the SSH session disconnects.
            # Phase 3: Tail logs from worker 0.
            # ----------------------------------------------------------------
            print("--- SETTING UP AND LAUNCHING ---")
            setup_launch_script = f"""
            $TPU_NAME = "{TPU_NAME}"
            $ZONE = "{ZONE}"
            $BUCKET = "{BUCKET}"

            # PHASE 1: Setup
            foreach ($i in 0..7) {{
                Write-Host "--- PREPARING WORKER $i ---" -ForegroundColor Cyan
                $cmd = "gcloud storage cp gs://$BUCKET/main.py . && pip install tokenizers numpy jax[tpu]==0.6.2 flax optax -f storage.googleapis.com && mkdir -p jax_cache"
                gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=$i --tunnel-through-iap --ssh-flag="-batch" --command=$cmd
            }}

            # PHASE 2: Launch
            Write-Host "--- LAUNCHING TRAINING ---" -ForegroundColor Green
            gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker="all" --tunnel-through-iap --ssh-flag="-batch" --command="nohup python3 -u main.py > v5.log 2>&1 &"

            Write-Host "--- CAT LOGS (worker 0) ---" -ForegroundColor Cyan
            echo "" | gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0 --tunnel-through-iap --command="cat v5.log"

            # PHASE 3: Tail
            Write-Host "--- TAILING LOGS ---" -ForegroundColor Cyan
            echo "" | gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=0 --tunnel-through-iap --command="stdbuf -oL tail -f v5.log"
            """
            run_ps(setup_launch_script)

        # ====================================================================
        # Reuse Existing TPU Pod Path
        # Kill any running processes, sync the new code, and relaunch.
        # No dependency reinstall needed — environment is already set up.
        # ====================================================================
        else:
            # Kill stale Python processes so the new run starts clean
            print("--- REUSING EXISTING TPU: KILLING OLD PROCESSES ---")
            kill_script = f"""
            $TPU_NAME = "{TPU_NAME}"
            $ZONE = "{ZONE}"

            foreach ($i in 0..7) {{
                Write-Host "--- FORCE KILLING PROCESSES ON WORKER $i ---" -ForegroundColor Cyan
                echo "" | gcloud alpha compute tpus tpu-vm ssh $TPU_NAME `
                    --zone=$ZONE `
                    --worker=$i `
                    --tunnel-through-iap `
                    --command="sudo pkill -9 -f python"
            }}

            Write-Host "DONE." -ForegroundColor Green
            """
            run_ps(kill_script)

            # Sync fresh code and relaunch
            print("--- SYNCING AND RELAUNCHING ON EXISTING TPU ---")
            relaunch_script = f"""
            $TPU_NAME = "{TPU_NAME}"
            $ZONE = "{ZONE}"
            $BUCKET = "{BUCKET}"

            foreach ($i in 0..7) {{
                Write-Host "--- UPDATING WORKER $i ---" -ForegroundColor Cyan
                $cmd = "rm -f main.py && rm -rf jax_cache && gcloud storage cp gs://$BUCKET/main.py . "
                gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=$i --tunnel-through-iap --ssh-flag="-batch" --command=$cmd
            }}

            Write-Host "--- LAUNCHING TRAINING ---" -ForegroundColor Green
            gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker="all" --tunnel-through-iap --ssh-flag="-batch" --command="nohup python3 -u main.py > v5.log 2>&1 &"

            Write-Host "--- CAT LOGS (worker 0) ---" -ForegroundColor Cyan
            echo "" | gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0 --tunnel-through-iap --command="cat v5.log"

            Write-Host "--- TAILING LOGS ---" -ForegroundColor Cyan
            echo "" | gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=0 --tunnel-through-iap --command="stdbuf -oL tail -f v5.log"
            """
            run_ps(relaunch_script)

    except Exception as e:
        print(f"Error occurred: {e}")

    # Wait for user confirmation before repeating the cycle
    try:
        input("\nTraining finished or crashed. Press Enter to REPEAT or Ctrl+C to stop.")
    except KeyboardInterrupt:
        break
