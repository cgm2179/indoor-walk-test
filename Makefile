# UNet Path-Loss Surrogate pipeline (spec §10: one command regenerates all)
#   make everything   = prepare + test + dataset + audit + assets
# Phase C (training) runs on Colab: SIM/phase_c_train_colab.ipynb

PY = python3

.PHONY: everything prepare test sample dataset audit assets optimizer model clean

# fetch the trained surrogate (124 MB, too big for git) from the GitHub release
model:
	gh release download surrogate-v1 -p pl_unet.onnx -O SIM/web/pl_unet.onnx \
		--repo cgm2179/indoor-walk-test --clobber

# fetch the 10k-sample training dataset (1.2 GB) instead of regenerating it
dataset-fetch:
	gh release download dataset-v1 -p 'shard_*.npz' -D SIM/dataset \
		--repo cgm2179/indoor-walk-test --clobber

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
