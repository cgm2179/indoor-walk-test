# UNet Path-Loss Surrogate pipeline (spec §10: one command regenerates all)
#   make everything   = prepare + test + dataset + audit + assets
# Phase C (training) runs on Colab: SIM/phase_c_train_colab.ipynb

PY = python3

.PHONY: everything prepare test sample dataset audit assets optimizer clean

everything: prepare test dataset audit assets

prepare:
	$(PY) SIM/phase_a.py --prepare

test:
	$(PY) SIM/phase_a.py --test
	$(PY) SIM/phase_d_calibrate.py --sanity

sample:
	$(PY) SIM/phase_a.py --sample

dataset:
	$(PY) SIM/phase_b_dataset.py

audit:
	$(PY) SIM/phase_b_dataset.py --audit

assets:
	$(PY) SIM/export_web_assets.py

optimizer:
	$(PY) SIM/phase_e_optimizer.py --objective coverage --stride 8 \
		--tx-power 23 --gain 3

clean:
	rm -rf SIM/preview SIM/dataset/shard_*.npz
