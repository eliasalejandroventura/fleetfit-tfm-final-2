"""Verify frozen artifact identity and independent inference after installation."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import importlib.metadata
import numpy as np
from load_factor_recommender import LoadFactorRecommender


def validate() -> dict:
    app = Path(__file__).resolve().parent
    root = Path(os.environ.get('TFM_MODEL_ARTIFACT_DIRECTORY', app.parent / '03_outputs/model')).resolve()
    contract = json.loads((root / 'FINAL_PREDICTION_CONTRACT.json').read_text())
    hashes = {}
    for role in ('pipeline', 'context'):
        path = root / contract[f'{role}_file']
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != contract[f'{role}_sha256']:
            raise RuntimeError(f'{role} differs from the distributed contract')
        hashes[path.name] = actual
    for requirement in (app / 'requirements.txt').read_text().splitlines():
        name, expected = requirement.split('==')
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f'Install the pinned application requirements: {name}=={expected}')
    reference = [('MCO', 'MIA', 699, 0.8056376576423645), ('JFK', 'MIA', 888, 0.8708228468894958), ('LGA', 'LAX', 625, 0.7830745577812195)]
    predictions=[]
    with LoadFactorRecommender(root) as model:
        if len(model.features) != 40:
            raise RuntimeError('Expected 40 predictors')
        booster = model.pipeline.steps[-1][1].get_booster()
        version = json.loads(booster.save_config())['version']
        if version != [3,4,1]:
            raise RuntimeError(f'Unexpected loaded XGBoost version: {version}')
        for origin,destination,aircraft,expected in reference:
            frame=model.score(origin,destination)
            value=float(frame.loc[frame.AIRCRAFT_TYPE.eq(aircraft),'PREDICTED_LF'].iloc[0])
            np.testing.assert_allclose(value,expected,rtol=0,atol=1e-7)
            predictions.append(dict(origin=origin,destination=destination,aircraft=aircraft,predicted_lf=value,candidates=len(frame)))
    return dict(status='PASS',artifact_hashes=hashes,features=40,reference_predictions=predictions)

if __name__ == '__main__':
    print(json.dumps(validate(),indent=2))
