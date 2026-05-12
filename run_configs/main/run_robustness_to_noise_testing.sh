# Experiment cookbook: probing cases where IL-ASGN can outperform vanilla backprop

# Shared conventions
# - All commands run from repo root unless noted
# - Metrics logs live under output/ilasgn_runs/ and output/backprop_runs/
# - Compare runs with: python scripts/analysis/benchmarking/backprop_vs_pc_vs_ilasgn.py --ilasgn <IL-ASGN json> --backprop <backprop json> [--pc <PC json>]

run_if_missing() {
  local target="$1"
  shift
  if [ -f "$target" ]; then
    echo "Skipping: $target already exists"
  else
    "$@"
  fi
}

############################
# 1) MNIST: wider models with light regularization
############################
# Hypothesis: IL-ASGN may retain accuracy with fewer optimization steps.
# Backprop baseline (PyTorch MLP aligned with IL-ASGN architecture)
run_if_missing output/backprop_runs/mnist_wide.json \
  python core/backprop_implementation.py \
    --dataset mnist --device auto --seed 0 \
    --width 512 --batch-size 150 --epochs 2 \
    --log-every 50 --eval-every 1 \
    --metrics-json output/backprop_runs/mnist_wide.json

# IL-ASGN counterpart (uses the adaptive IL-ASGN trainer)
run_if_missing output/ilasgn_runs/mnist_wide.json \
  python core/train_path_integral_pc.py \
    --dataset mnist --device auto --seed 0 \
    --width 512 --train-steps 800 \
    --eval-every 20 --log-every 20 --lambda-cap 1 \
    --dt 0.006 --rollout-steps 24 --rollout-burnin 4 --rollout-tol 2e-4 \
    --metrics-json output/ilasgn_runs/mnist_wide.json

############################
# 2) MNIST: label noise stress test
############################
# Hypothesis: IL-ASGN resilience under heavy noise.
# Backprop baseline (label corruption now supported)
run_if_missing output/backprop_runs/mnist_noisy.json \
  python core/backprop_implementation.py \
    --dataset mnist --device auto --seed 0 \
    --width 256 --batch-size 100 --epochs 1 \
    --label-noise 0.3 --eval-every 1 --log-every 50 \
    --metrics-json output/backprop_runs/mnist_noisy.json

# IL-ASGN noisy run
run_if_missing output/ilasgn_runs/mnist_noisy.json \
  python core/train_path_integral_pc.py \
    --dataset mnist --device auto --seed 0 \
    --width 256 --train-steps 600 \
    --label-noise 0.3 --eval-every 20 --log-every 20 --lambda-cap 1 \
    --dt 0.006 --rollout-steps 20 --rollout-burnin 3 --rollout-tol 2e-4 \
    --metrics-json output/ilasgn_runs/mnist_noisy.json

############################
# 3) Reporting helper
############################
# Summarize the most promising cases side by side once logs are ready (see comparison printers below).

############################
# 4) Input-noise sweeps (Gaussian σ and salt-pepper)
############################
# Loop over Gaussian std values for all three methods. σ is specified with --input-noise-level for the backprop and PC
# runs, and --input-noise-std for the IL-ASGN trainer. Each run repeats for seeds 0–9 with seed-suffixed metrics.
for seed in 0 1 2 3 4 5 6 7 8 9; do
  for sigma in 0 0.1 0.2 0.3 0.4 0.5; do
    # Backprop baseline
    run_if_missing output/backprop_runs/mnist_gaussian_sigma${sigma}_seed${seed}.json \
      python core/backprop_implementation.py \
        --dataset mnist --device auto --seed ${seed} \
        --width 256 --batch-size 150 --epochs 1 \
        --input-noise-type gaussian --input-noise-level ${sigma} \
        --metrics-json output/backprop_runs/mnist_gaussian_sigma${sigma}_seed${seed}.json

    # Predictive coding baseline
    run_if_missing output/pc_runs/mnist_gaussian_sigma${sigma}_seed${seed}.json \
      python core/pc_mnist_baseline.py \
        --dataset mnist --device auto --seed ${seed} \
        --width 256 --depth 3 --train-steps 400 \
        --input-noise-type gaussian --input-noise-level ${sigma} \
        --metrics-json output/pc_runs/mnist_gaussian_sigma${sigma}_seed${seed}.json
  done

  for sigma in 0 0.1 0.2 0.3 0.4 0.5; do
    run_if_missing output/ilasgn_runs/mnist_gaussian_sigma${sigma}_seed${seed}.json \
      python core/train_path_integral_pc.py \
        --dataset mnist --device auto --seed ${seed} \
        --width 256 --train-steps 400 \
        --input-noise-type gaussian --input-noise-std ${sigma} \
        --eval-every 20 --log-every 20 --lambda-cap 1 \
        --dt 0.006 --rollout-steps 20 --rollout-burnin 3 --rollout-tol 2e-4 \
        --metrics-json output/ilasgn_runs/mnist_gaussian_sigma${sigma}_seed${seed}.json
  done

  # Salt-and-pepper sweeps replace a fraction of pixels with 0/1 values. Probabilities 0.1/0.2/0.3 correspond to 10%/20%/30%.
  for prob in 0.1 0.2 0.3; do
    # Backprop baseline
    run_if_missing output/backprop_runs/mnist_saltpepper_${prob}_seed${seed}.json \
      python core/backprop_implementation.py \
        --dataset mnist --device auto --seed ${seed} \
        --width 256 --batch-size 150 --epochs 1 \
        --input-noise-type salt_pepper --input-noise-level ${prob} \
        --metrics-json output/backprop_runs/mnist_saltpepper_${prob}_seed${seed}.json

    # Predictive coding baseline
    run_if_missing output/pc_runs/mnist_saltpepper_${prob}_seed${seed}.json \
      python core/pc_mnist_baseline.py \
        --dataset mnist --device auto --seed ${seed} \
        --width 256 --depth 3 --train-steps 400 \
        --input-noise-type salt_pepper --input-noise-level ${prob} \
        --metrics-json output/pc_runs/mnist_saltpepper_${prob}_seed${seed}.json
  done

  for prob in 0.1 0.2 0.3; do
    run_if_missing output/ilasgn_runs/mnist_saltpepper_${prob}_seed${seed}.json \
      python core/train_path_integral_pc.py \
        --dataset mnist --device auto --seed ${seed} \
        --width 256 --train-steps 400 \
        --input-noise-type salt_pepper --salt-pepper-prob ${prob} \
        --eval-every 20 --log-every 20 --lambda-cap 1 \
        --dt 0.006 --rollout-steps 20 --rollout-burnin 3 --rollout-tol 2e-4 \
        --metrics-json output/ilasgn_runs/mnist_saltpepper_${prob}_seed${seed}.json
  done
done

############################
# 5) Occlusion block noise (MNIST, vanilla PC vs backprop vs IL-ASGN)
############################
# Sweep three occlusion severities (block side as a fraction of the image size) across
# the vanilla predictive coding baseline (JPC/JAX), a standard backprop MLP, and IL-ASGN.
# Budgets are matched at 400 update steps for clean comparisons across methods.
for seed in 0 1 2 3 4 5 6 7 8 9; do
  for frac in 0.1 0.2 0.3; do
    # Vanilla predictive coding with JPC/JAX
    run_if_missing output/pc_runs/mnist_occlusion_${frac}_seed${seed}.json \
      python core/pc_mnist_baseline.py \
        --dataset mnist --device auto --seed ${seed} \
        --width 256 --depth 3 --train-steps 400 \
        --input-noise-type occlusion --input-noise-level ${frac} \
        --metrics-json output/pc_runs/mnist_occlusion_${frac}_seed${seed}.json

    # Backprop baseline (matches data corruption + architecture family)
    run_if_missing output/backprop_runs/mnist_occlusion_${frac}_seed${seed}.json \
      python core/backprop_implementation.py \
        --dataset mnist --device auto --seed ${seed} \
        --width 256 --batch-size 150 --epochs 1 \
        --input-noise-type occlusion --input-noise-level ${frac} \
        --metrics-json output/backprop_runs/mnist_occlusion_${frac}_seed${seed}.json

    # IL-ASGN with occlusion-corrupted inputs
    run_if_missing output/ilasgn_runs/mnist_occlusion_${frac}_seed${seed}.json \
      python core/train_path_integral_pc.py \
        --dataset mnist --device auto --seed ${seed} \
        --width 256 --train-steps 400 \
        --input-noise-type occlusion --occlusion-fraction ${frac} \
        --eval-every 20 --log-every 20 --lambda-cap 1 \
        --dt 0.006 --rollout-steps 20 --rollout-burnin 3 --rollout-tol 2e-4 \
        --metrics-json output/ilasgn_runs/mnist_occlusion_${frac}_seed${seed}.json
  done
done

# Combined robustness plot (Gaussian, salt-pepper, and occlusion panels; includes
# sigma=0) plus mean-accuracy summaries across each sweep. Run after the occlusion
# loop above so all panels are populated. Pass --metrics-root <repo path> if
# running from outside the repository root (e.g., IDE run configs) so the script
# can find the logs.
run_if_missing output/pi_plots_out/mnist_noise_robustness.png \
  python scripts/plotting/benchmarking/plot_noise_robustness.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --gaussian-ilasgn output/ilasgn_runs/mnist_gaussian_sigma{level}_seed{seed}.json \
    --gaussian-backprop output/backprop_runs/mnist_gaussian_sigma{level}_seed{seed}.json \
    --gaussian-pc output/pc_runs/mnist_gaussian_sigma{level}_seed{seed}.json \
    --saltpepper-ilasgn output/ilasgn_runs/mnist_saltpepper_{level}_seed{seed}.json \
    --saltpepper-backprop output/backprop_runs/mnist_saltpepper_{level}_seed{seed}.json \
    --saltpepper-pc output/pc_runs/mnist_saltpepper_{level}_seed{seed}.json \
    --occlusion-ilasgn output/ilasgn_runs/mnist_occlusion_{level}_seed{seed}.json \
    --occlusion-backprop output/backprop_runs/mnist_occlusion_{level}_seed{seed}.json \
    --occlusion-pc output/pc_runs/mnist_occlusion_{level}_seed{seed}.json \
    --output output/pi_plots_out/mnist_noise_robustness.png

############################
# Comparison printers (run after all logs above are produced)
############################
# MNIST comparisons
# Each invocation prints a single block then exits; seeing one block in the terminal
# means the script finished normally.
python scripts/analysis/benchmarking/backprop_vs_pc_vs_ilasgn.py \
  --ilasgn output/ilasgn_runs/mnist_wide.json \
  --backprop output/backprop_runs/mnist_wide.json

python scripts/analysis/benchmarking/backprop_vs_pc_vs_ilasgn.py \
  --ilasgn output/ilasgn_runs/mnist_noisy.json \
  --backprop output/backprop_runs/mnist_noisy.json

# Occlusion sweep printers (drop the --pc line if you skipped the PC runs)
for seed in 0 1 2 3 4 5 6 7 8 9; do
  for frac in 0.1 0.2 0.3; do
    python scripts/analysis/benchmarking/backprop_vs_pc_vs_ilasgn.py \
      --ilasgn output/ilasgn_runs/mnist_occlusion_${frac}_seed${seed}.json \
      --backprop output/backprop_runs/mnist_occlusion_${frac}_seed${seed}.json \
      --pc output/pc_runs/mnist_occlusion_${frac}_seed${seed}.json
  done

  # Gaussian sweep printers
  for sigma in 0 0.1 0.2 0.3 0.4 0.5; do
    python scripts/analysis/benchmarking/backprop_vs_pc_vs_ilasgn.py \
      --ilasgn output/ilasgn_runs/mnist_gaussian_sigma${sigma}_seed${seed}.json \
      --backprop output/backprop_runs/mnist_gaussian_sigma${sigma}_seed${seed}.json \
      --pc output/pc_runs/mnist_gaussian_sigma${sigma}_seed${seed}.json
  done

  # Salt-and-pepper sweep printers
  for prob in 0.1 0.2 0.3; do
    python scripts/analysis/benchmarking/backprop_vs_pc_vs_ilasgn.py \
      --ilasgn output/ilasgn_runs/mnist_saltpepper_${prob}_seed${seed}.json \
      --backprop output/backprop_runs/mnist_saltpepper_${prob}_seed${seed}.json \
      --pc output/pc_runs/mnist_saltpepper_${prob}_seed${seed}.json
  done
done

############################
# 6) CIFAR-10 input-noise sweeps (Gaussian, salt-pepper, occlusion)
############################
# Mirror the MNIST noise sweeps for CIFAR-10 using the backprop, predictive coding, and
# IL-ASGN trainers. Metrics land in output/backprop_runs/, output/pc_runs/, and output/ilasgn_runs/ with
# noise-specific filenames. Budgets use the same small-model configuration that showed
# the biggest IL-ASGN lift (width 128, batch ~133 for backprop vs. 1.5k IL-ASGN steps) so
# robustness comparisons reflect that capacity-limited setting instead of the wider 512-
# hidden model.

# Gaussian sweeps (σ = 0 keeps a clean control run in the same series)
for seed in 0 1 2 3 4 5 6 7 8 9; do
  for sigma in 0 0.1 0.2 0.3 0.4 0.5; do
    # Backprop CIFAR-10 baseline
    run_if_missing output/backprop_runs/cifar10_gaussian_sigma${sigma}_seed${seed}.json \
      python core/backprop_implementation.py \
        --dataset cifar10 --augment \
        --width 128 --batch-size 133 --lr 1e-3 \
        --epochs 4 --eval-every 1 --log-every 100 \
        --input-noise-type gaussian --input-noise-level ${sigma} \
        --metrics-json output/backprop_runs/cifar10_gaussian_sigma${sigma}_seed${seed}.json

    # Predictive coding CIFAR-10 baseline (JPC/JAX)
    run_if_missing output/pc_runs/cifar10_gaussian_sigma${sigma}_seed${seed}.json \
      python core/pc_cifar10_baseline.py \
        --device auto --seed ${seed} \
        --width 128 --depth 3 --train-steps 1500 \
        --augment --input-noise-type gaussian --input-noise-level ${sigma} \
        --metrics-json output/pc_runs/cifar10_gaussian_sigma${sigma}_seed${seed}.json

    # IL-ASGN CIFAR-10 counterpart
    run_if_missing output/ilasgn_runs/cifar10_gaussian_sigma${sigma}_seed${seed}.json \
      python core/train_path_integral_pc.py \
        --dataset cifar10 --augment \
        --width 128 --dt 0.004 --train-steps 1500 \
        --input-noise-type gaussian --input-noise-std ${sigma} \
        --eval-every 30 --log-every 30 --lambda-cap 1 \
        --rollout-steps 16 --rollout-burnin 3 --rollout-tol 3e-4 \
        --metrics-json output/ilasgn_runs/cifar10_gaussian_sigma${sigma}_seed${seed}.json
  done

  # Salt-and-pepper sweeps (probability is the total flip chance across salt+pepper)
  for prob in 0.1 0.2 0.3; do
    # Backprop CIFAR-10 baseline
    run_if_missing output/backprop_runs/cifar10_saltpepper_${prob}_seed${seed}.json \
      python core/backprop_implementation.py \
        --dataset cifar10 --augment \
        --width 128 --batch-size 133 --lr 1e-3 \
        --epochs 4 --eval-every 1 --log-every 100 \
        --input-noise-type salt_pepper --input-noise-level ${prob} \
        --metrics-json output/backprop_runs/cifar10_saltpepper_${prob}_seed${seed}.json

    # Predictive coding CIFAR-10 baseline (JPC/JAX)
    run_if_missing output/pc_runs/cifar10_saltpepper_${prob}_seed${seed}.json \
      python core/pc_cifar10_baseline.py \
        --device auto --seed ${seed} \
        --width 128 --depth 3 --train-steps 1500 \
        --augment --input-noise-type salt_pepper --input-noise-level ${prob} \
        --metrics-json output/pc_runs/cifar10_saltpepper_${prob}_seed${seed}.json

    # IL-ASGN CIFAR-10 counterpart
    run_if_missing output/ilasgn_runs/cifar10_saltpepper_${prob}_seed${seed}.json \
      python core/train_path_integral_pc.py \
        --dataset cifar10 --augment \
        --width 128 --dt 0.004 --train-steps 1500 \
        --input-noise-type salt_pepper --salt-pepper-prob ${prob} \
        --eval-every 30 --log-every 30 --lambda-cap 1 \
        --rollout-steps 16 --rollout-burnin 3 --rollout-tol 3e-4 \
        --metrics-json output/ilasgn_runs/cifar10_saltpepper_${prob}_seed${seed}.json
  done

  # Occlusion sweeps (block side fraction of the 32×32 image)
  for frac in 0.1 0.2 0.3; do
    # Backprop CIFAR-10 baseline
    run_if_missing output/backprop_runs/cifar10_occlusion_${frac}_seed${seed}.json \
      python core/backprop_implementation.py \
        --dataset cifar10 --augment \
        --width 128 --batch-size 133 --lr 1e-3 \
        --epochs 4 --eval-every 1 --log-every 100 \
        --input-noise-type occlusion --input-noise-level ${frac} \
        --metrics-json output/backprop_runs/cifar10_occlusion_${frac}_seed${seed}.json

    # Predictive coding CIFAR-10 baseline (JPC/JAX)
    run_if_missing output/pc_runs/cifar10_occlusion_${frac}_seed${seed}.json \
      python core/pc_cifar10_baseline.py \
        --device auto --seed ${seed} \
        --width 128 --depth 3 --train-steps 1500 \
        --augment --input-noise-type occlusion --input-noise-level ${frac} \
        --metrics-json output/pc_runs/cifar10_occlusion_${frac}_seed${seed}.json

    # IL-ASGN CIFAR-10 counterpart
    run_if_missing output/ilasgn_runs/cifar10_occlusion_${frac}_seed${seed}.json \
      python core/train_path_integral_pc.py \
        --dataset cifar10 --augment \
        --width 128 --dt 0.004 --train-steps 1500 \
        --input-noise-type occlusion --occlusion-fraction ${frac} \
        --eval-every 30 --log-every 30 --lambda-cap 1 \
        --rollout-steps 16 --rollout-burnin 3 --rollout-tol 3e-4 \
        --metrics-json output/ilasgn_runs/cifar10_occlusion_${frac}_seed${seed}.json
  done
done

############################
# CIFAR-10 noise comparisons and plotting (run after sweeps finish)
############################
  # Print CIFAR-10 IL-ASGN vs backprop results for each noise family, then generate
  # a CIFAR-10 robustness plot analogous to the MNIST figure (Gaussian, salt-pepper,
  # occlusion panels).
for seed in 0 1 2 3 4 5 6 7 8 9; do
  for sigma in 0 0.1 0.2 0.3 0.4 0.5; do
    python scripts/analysis/benchmarking/backprop_vs_pc_vs_ilasgn.py \
      --ilasgn output/ilasgn_runs/cifar10_gaussian_sigma${sigma}_seed${seed}.json \
      --backprop output/backprop_runs/cifar10_gaussian_sigma${sigma}_seed${seed}.json \
      --pc output/pc_runs/cifar10_gaussian_sigma${sigma}_seed${seed}.json
  done

  for prob in 0.1 0.2 0.3; do
    python scripts/analysis/benchmarking/backprop_vs_pc_vs_ilasgn.py \
      --ilasgn output/ilasgn_runs/cifar10_saltpepper_${prob}_seed${seed}.json \
      --backprop output/backprop_runs/cifar10_saltpepper_${prob}_seed${seed}.json \
      --pc output/pc_runs/cifar10_saltpepper_${prob}_seed${seed}.json
  done

  for frac in 0.1 0.2 0.3; do
    python scripts/analysis/benchmarking/backprop_vs_pc_vs_ilasgn.py \
      --ilasgn output/ilasgn_runs/cifar10_occlusion_${frac}_seed${seed}.json \
      --backprop output/backprop_runs/cifar10_occlusion_${frac}_seed${seed}.json \
      --pc output/pc_runs/cifar10_occlusion_${frac}_seed${seed}.json
  done
done

  run_if_missing output/pi_plots_out/cifar10_noise_robustness.png \
    python scripts/plotting/benchmarking/plot_noise_robustness.py \
      --metrics-root . \
      --seeds 0 1 2 3 4 5 6 7 8 9 \
      --gaussian-ilasgn output/ilasgn_runs/cifar10_gaussian_sigma{level}_seed{seed}.json \
      --gaussian-backprop output/backprop_runs/cifar10_gaussian_sigma{level}_seed{seed}.json \
      --gaussian-levels 0 0.1 0.2 0.3 0.4 0.5 \
      --saltpepper-levels 0.1 0.2 0.3 \
      --saltpepper-ilasgn output/ilasgn_runs/cifar10_saltpepper_{level}_seed{seed}.json \
      --saltpepper-backprop output/backprop_runs/cifar10_saltpepper_{level}_seed{seed}.json \
      --occlusion-levels 0.1 0.2 0.3 \
      --occlusion-ilasgn output/ilasgn_runs/cifar10_occlusion_{level}_seed{seed}.json \
      --occlusion-backprop output/backprop_runs/cifar10_occlusion_{level}_seed{seed}.json \
      --output output/pi_plots_out/cifar10_noise_robustness.png

############################
# 8) MNIST bottleneck input-noise sweeps (Gaussian, salt-pepper, occlusion)
############################
# Mirror the CIFAR-10 and wider-model MNIST noise sweeps on the narrow MNIST model (width 128).
# Metrics include a "mnist_bottleneck_*" prefix so the noisy runs are easy to find next to the
# clean bottleneck baselines above.

for seed in 0 1 2 3 4 5 6 7 8 9; do
  # Gaussian sweeps (reuse σ grid from the wider-model MNIST runs)
  for sigma in 0 0.1 0.2 0.3 0.4 0.5; do
    # Backprop bottleneck baseline under Gaussian corruption
    run_if_missing output/backprop_runs/mnist_bottleneck_gaussian_sigma${sigma}_seed${seed}.json \
      python core/backprop_implementation.py \
        --dataset mnist --device auto --seed ${seed} \
        --width 128 --batch-size 150 --epochs 2 \
        --input-noise-type gaussian --input-noise-level ${sigma} \
        --log-every 50 --eval-every 1 \
        --metrics-json output/backprop_runs/mnist_bottleneck_gaussian_sigma${sigma}_seed${seed}.json

    # Predictive coding bottleneck counterpart (JPC/JAX)
    run_if_missing output/pc_runs/mnist_bottleneck_gaussian_sigma${sigma}_seed${seed}.json \
      python core/pc_mnist_baseline.py \
        --dataset mnist --device auto --seed ${seed} \
        --preset mnist_bottleneck --depth 3 --train-steps 400 \
        --input-noise-type gaussian --input-noise-level ${sigma} \
        --metrics-json output/pc_runs/mnist_bottleneck_gaussian_sigma${sigma}_seed${seed}.json

    # IL-ASGN bottleneck counterpart
    run_if_missing output/ilasgn_runs/mnist_bottleneck_gaussian_sigma${sigma}_seed${seed}.json \
      python core/train_path_integral_pc.py \
        --dataset mnist --device auto --seed ${seed} \
        --width 128 --train-steps 800 \
        --input-noise-type gaussian --input-noise-std ${sigma} \
        --eval-every 20 --log-every 20 --lambda-cap 1 \
        --dt 0.006 --rollout-steps 24 --rollout-burnin 4 --rollout-tol 2e-4 \
        --metrics-json output/ilasgn_runs/mnist_bottleneck_gaussian_sigma${sigma}_seed${seed}.json
  done

  # Salt-and-pepper sweeps (flip probability matches the wider-model MNIST experiment)
  for prob in 0.1 0.2 0.3; do
    # Backprop bottleneck baseline
    run_if_missing output/backprop_runs/mnist_bottleneck_saltpepper_${prob}_seed${seed}.json \
      python core/backprop_implementation.py \
        --dataset mnist --device auto --seed ${seed} \
        --width 128 --batch-size 150 --epochs 2 \
        --input-noise-type salt_pepper --input-noise-level ${prob} \
        --log-every 50 --eval-every 1 \
        --metrics-json output/backprop_runs/mnist_bottleneck_saltpepper_${prob}_seed${seed}.json

    # Predictive coding bottleneck counterpart (JPC/JAX)
    run_if_missing output/pc_runs/mnist_bottleneck_saltpepper_${prob}_seed${seed}.json \
      python core/pc_mnist_baseline.py \
        --dataset mnist --device auto --seed ${seed} \
        --preset mnist_bottleneck --depth 3 --train-steps 400 \
        --input-noise-type salt_pepper --input-noise-level ${prob} \
        --metrics-json output/pc_runs/mnist_bottleneck_saltpepper_${prob}_seed${seed}.json

    # IL-ASGN bottleneck counterpart
    run_if_missing output/ilasgn_runs/mnist_bottleneck_saltpepper_${prob}_seed${seed}.json \
      python core/train_path_integral_pc.py \
        --dataset mnist --device auto --seed ${seed} \
        --width 128 --train-steps 800 \
        --input-noise-type salt_pepper --salt-pepper-prob ${prob} \
        --eval-every 20 --log-every 20 --lambda-cap 1 \
        --dt 0.006 --rollout-steps 24 --rollout-burnin 4 --rollout-tol 2e-4 \
        --metrics-json output/ilasgn_runs/mnist_bottleneck_saltpepper_${prob}_seed${seed}.json
  done

  # Occlusion sweeps (block side as a fraction of the 28×28 MNIST image)
  for frac in 0.1 0.2 0.3; do
    # Backprop bottleneck baseline
    run_if_missing output/backprop_runs/mnist_bottleneck_occlusion_${frac}_seed${seed}.json \
      python core/backprop_implementation.py \
        --dataset mnist --device auto --seed ${seed} \
        --width 128 --batch-size 150 --epochs 2 \
        --input-noise-type occlusion --input-noise-level ${frac} \
        --log-every 50 --eval-every 1 \
        --metrics-json output/backprop_runs/mnist_bottleneck_occlusion_${frac}_seed${seed}.json

    # Predictive coding bottleneck counterpart (JPC/JAX)
    run_if_missing output/pc_runs/mnist_bottleneck_occlusion_${frac}_seed${seed}.json \
      python core/pc_mnist_baseline.py \
        --dataset mnist --device auto --seed ${seed} \
        --preset mnist_bottleneck --depth 3 --train-steps 400 \
        --input-noise-type occlusion --input-noise-level ${frac} \
        --metrics-json output/pc_runs/mnist_bottleneck_occlusion_${frac}_seed${seed}.json

    # IL-ASGN bottleneck counterpart
    run_if_missing output/ilasgn_runs/mnist_bottleneck_occlusion_${frac}_seed${seed}.json \
      python core/train_path_integral_pc.py \
        --dataset mnist --device auto --seed ${seed} \
        --width 128 --train-steps 800 \
        --input-noise-type occlusion --occlusion-fraction ${frac} \
        --eval-every 20 --log-every 20 --lambda-cap 1 \
        --dt 0.006 --rollout-steps 24 --rollout-burnin 4 --rollout-tol 2e-4 \
        --metrics-json output/ilasgn_runs/mnist_bottleneck_occlusion_${frac}_seed${seed}.json
  done
done

# Bottleneck robustness plotting (Gaussian, salt-pepper, occlusion panels)
# Run after the loops above finish so all panels populate.
run_if_missing output/pi_plots_out/mnist_bottleneck_noise_robustness.png \
  python scripts/plotting/benchmarking/plot_noise_robustness.py \
    --metrics-root . \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --gaussian-ilasgn output/ilasgn_runs/mnist_bottleneck_gaussian_sigma{level}_seed{seed}.json \
    --gaussian-backprop output/backprop_runs/mnist_bottleneck_gaussian_sigma{level}_seed{seed}.json \
    --saltpepper-ilasgn output/ilasgn_runs/mnist_bottleneck_saltpepper_{level}_seed{seed}.json \
    --saltpepper-backprop output/backprop_runs/mnist_bottleneck_saltpepper_{level}_seed{seed}.json \
    --occlusion-levels 0.1 0.2 0.3 \
    --occlusion-ilasgn output/ilasgn_runs/mnist_bottleneck_occlusion_{level}_seed{seed}.json \
    --occlusion-backprop output/backprop_runs/mnist_bottleneck_occlusion_{level}_seed{seed}.json \
    --output output/pi_plots_out/mnist_bottleneck_noise_robustness.png

# Bottleneck noise comparisons (per corruption family, mirrors CIFAR-10/MNIST printers)
for seed in 0 1 2 3 4 5 6 7 8 9; do
  for sigma in 0 0.1 0.2 0.3 0.4 0.5; do
    python scripts/analysis/benchmarking/backprop_vs_pc_vs_ilasgn.py \
      --ilasgn output/ilasgn_runs/mnist_bottleneck_gaussian_sigma${sigma}_seed${seed}.json \
      --backprop output/backprop_runs/mnist_bottleneck_gaussian_sigma${sigma}_seed${seed}.json
  done

  for prob in 0.1 0.2 0.3; do
    python scripts/analysis/benchmarking/backprop_vs_pc_vs_ilasgn.py \
      --ilasgn output/ilasgn_runs/mnist_bottleneck_saltpepper_${prob}_seed${seed}.json \
      --backprop output/backprop_runs/mnist_bottleneck_saltpepper_${prob}_seed${seed}.json
  done

  for frac in 0.1 0.2 0.3; do
    python scripts/analysis/benchmarking/backprop_vs_pc_vs_ilasgn.py \
      --ilasgn output/ilasgn_runs/mnist_bottleneck_occlusion_${frac}_seed${seed}.json \
      --backprop output/backprop_runs/mnist_bottleneck_occlusion_${frac}_seed${seed}.json
  done
done
