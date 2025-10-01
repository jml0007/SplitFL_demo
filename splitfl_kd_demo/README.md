# Split Federated Learning (Demo)

## Install
```bash
pip install -r requirements.txt
```

## Quick demo
```bash
python -m src.main --config configs/default.yaml --fast_dev_run
```

## Structure
```
splitfl_kd_demo/
├─ configs/default.yaml
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ utils.py
│  ├─ data.py
│  ├─ models.py
│  ├─ client.py
│  ├─ fl_system.py
│  └─ main.py
├─ tests/test_shapes.py
├─ scripts/run_fast_demo.sh
├─ requirements.txt
├─ LICENSE
└─ README.md
```
