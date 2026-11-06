PY ?= .venv/bin/python
MODAL ?= PYTHONPATH=. .venv/bin/modal

setup:            ## create the venv and install everything
	uv venv --python 3.12 .venv && uv pip install --python $(PY) -r requirements.txt

test:             ## unit tests
	$(PY) -m pytest -q tests

app:              ## run the web app locally on :7860
	$(PY) app.py

docker:           ## build + run the app container on :7860
	docker build -t hybrid-isp . && docker run --rm -p 7860:7860 hybrid-isp

data:             ## build the SIDD dataset on Modal, then render ground truth + traditional baseline
	cd modal_app && ../.venv/bin/modal run prepare_data.py
	$(MODAL) run modal_app/render_gt.py

train:            ## train all models on Modal (teacher first; students need the teacher)
	$(MODAL) run modal_app/train.py --run teacher --minutes 165
	$(MODAL) run modal_app/train.py --run e2e --minutes 135
	$(MODAL) run modal_app/train.py --run nosigma --minutes 85
	$(MODAL) run modal_app/train.py --run student_kd --minutes 120
	$(MODAL) run modal_app/train.py --run student_scratch --minutes 90

weights:          ## export + quantise trained checkpoints (runs/ckpt/*.pt) into weights/
	$(PY) -m optimize.export_onnx runs/ckpt/teacher.pt weights/teacher_fp32.onnx
	$(PY) -m optimize.export_onnx runs/ckpt/e2e.pt weights/e2e_fp32.onnx
	$(PY) -m optimize.export_onnx runs/ckpt/student_kd.pt weights/student_fp32.onnx
	$(PY) -m optimize.quantize weights/student_fp32.onnx weights/student_int8.onnx

benchmark:        ## size / latency / memory table
	$(PY) -m optimize.benchmark weights/*.onnx --out runs/benchmark.json

eval:             ## full evaluation on the test split (Modal GPU) -> eval_P2_T77.md
	$(MODAL) run modal_app/evaluate.py
	$(PY) -m eval.report --team T77 --usns "1RUA24CSE0162,1RUA24CSE0150,1RUA24CSE0187,1RUA24CSE0167"

.PHONY: setup test app docker data train weights benchmark eval
