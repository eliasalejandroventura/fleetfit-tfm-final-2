"""Regression coverage for route geometry delivered to the map renderer."""
import json
from pathlib import Path
import unittest
import numpy as np
from streamlit.testing.v1 import AppTest

class RouteMapTests(unittest.TestCase):
    def test_map_connects_both_airports_across_the_dateline(self):
        for origin, destination in [('HNL', 'MEL'), ('LAX', 'NRT'), ('JFK', 'MIA')]:
            with self.subTest(route=f'{origin}-{destination}'):
                app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=120)
                app.session_state['submitted_route'] = {'origin': origin, 'destination': destination}
                app.run()
                self.assertFalse(app.exception)
                deck = json.loads(app.get('deck_gl_json_chart')[0].proto.json)
                layers = {layer['@@type']: layer for layer in deck['layers']}
                line = layers['PathLayer']
                path = np.asarray(line['data'][0]['path'])
                markers = layers['ScatterplotLayer']['data']
                self.assertTrue(np.isfinite(path).all())
                np.testing.assert_allclose(path[0], markers[0]['coordinates'], atol=1e-8)
                np.testing.assert_allclose(path[-1], markers[1]['coordinates'], atol=1e-8)
                self.assertLess(np.abs(np.diff(path[:, 0])).max(), 180)
                self.assertGreater(line['widthMinPixels'], 0)
                self.assertNotEqual(line.get('widthUnits'), '@@=pixels')
                self.assertLessEqual(abs(deck['initialViewState']['longitude']), 180)
                if origin == 'HNL':
                    self.assertGreater(deck['initialViewState']['longitude'], 140)
