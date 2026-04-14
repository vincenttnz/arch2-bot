# hyperopt_tuner.py
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import os
import yaml
from ultralytics import YOLO

class YOLOTuner:
    def __init__(self, data_yaml, base_weights, device=0):
        self.data_yaml = data_yaml
        self.base_weights = base_weights
        self.device = device

    def objective(self, params):
        # Params: lr0, lrf, momentum, weight_decay, box, cls, hsv_h, hsv_s, hsv_v, scale, fl_gamma
        model = YOLO(self.base_weights)
        try:
            results = model.train(
                data=self.data_yaml,
                epochs=10,  # short trial
                imgsz=640,
                device=self.device,
                workers=0,
                lr0=params['lr0'],
                lrf=params['lrf'],
                momentum=params['momentum'],
                weight_decay=params['weight_decay'],
                box=params['box'],
                cls=params['cls'],
                hsv_h=params['hsv_h'],
                hsv_s=params['hsv_s'],
                hsv_v=params['hsv_v'],
                scale=params['scale'],
                fl_gamma=params['fl_gamma'],
                verbose=False,
                exist_ok=True,
                name='tune_trial'
            )
            # Get best validation metric (mAP50-95)
            best_metric = results.results_dict['metrics/mAP50-95(B)']
        except Exception as e:
            print(f"Trial failed: {e}")
            return {'loss': 1.0, 'status': STATUS_OK}
        return {'loss': -best_metric, 'status': STATUS_OK}

    def tune(self, max_evals=20):
        space = {
            'lr0': hp.loguniform('lr0', np.log(1e-4), np.log(1e-2)),
            'lrf': hp.loguniform('lrf', np.log(1e-3), np.log(1e-1)),
            'momentum': hp.uniform('momentum', 0.8, 0.98),
            'weight_decay': hp.loguniform('weight_decay', np.log(1e-5), np.log(1e-3)),
            'box': hp.uniform('box', 5.0, 12.0),
            'cls': hp.uniform('cls', 0.2, 1.0),
            'hsv_h': hp.uniform('hsv_h', 0.0, 0.1),
            'hsv_s': hp.uniform('hsv_s', 0.0, 1.0),
            'hsv_v': hp.uniform('hsv_v', 0.0, 0.6),
            'scale': hp.uniform('scale', 0.0, 0.9),
            'fl_gamma': hp.uniform('fl_gamma', 0.0, 2.0),
        }
        trials = Trials()
        best = fmin(self.objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        return best