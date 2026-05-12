python scripts/plotting/pc_testing/plot_pc_equilibrium.py \
  --runner core/train_path_integral_pc.py \
  --device gpu \
  --seed 0 1 \
  --lambda-cap 0 1 \
  --train-steps 1000 \
  --eval-every 20 \
  --oracle-every 1 3 5 8 \
  --grid-dt 0.0036 0.005 0.007 \
  --grid-tol 5e-4 2e-4 1e-4 \
  --grid-steps 8 16 32 \
  --extra "--dataset mnist --augment --label-noise 0.0 --width 1024"


#python scripts/plotting/pc_testing/plot_lambda_gate_vs_pi.py \
#  --runner core/train_path_integral_pc.py \
#  --device gpu \
#  --seed 0 1 \
#  --train-steps 1000 \
#  --eval-every 20 \
#  --oracle-every 1 3 5 8 \
#  --grid-dt 0.0036 0.005 \
#  --grid-tol 1e-4 2e-4 5e-4 \
#  --grid-steps 8 16 32 \
#  --extra "--dataset mnist --augment --label-noise 0.0 --width 1024"

#python scripts/analysis/pc_testing/compare_lambda_caps.py \
#  --lam0 output/pi_plots_lambda0/pi_equilibration_timeseries.csv \
#  --lam1 output/pi_plots_lambda1/pi_equilibration_timeseries.csv \
#  --outdir compare_out
