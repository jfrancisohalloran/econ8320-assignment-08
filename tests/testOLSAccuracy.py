import unittest
import json
import pandas as pd
import numpy as np
import sys as system
import io
import re


with open("Lesson.ipynb", "r") as file:
    f_str = file.read()

doc = json.loads(f_str)

code = [i for i in doc['cells'] if i['cell_type']=='code']
si = {}
for i in code:
    for j in i['source']:
        if "#si-exercise" in j:
            exec(compile("".join(i['source']), '<string>', 'exec'))



# todo: replace this with an actual test
class TestCase(unittest.TestCase):

    def testOLSAccurate(self):
        data = pd.read_csv("tests/files/assignment8Data.csv")
        x = data[['sex', 'age', 'educ', 'white']]
        y = data['incwage']
        reg = RegressionModel(x, y, create_intercept=True)
        reg.ols_regression()

        expected_results = {
            'sex': {'coefficient': -13565.410626563069, 'standard_error': 584.9764665087288,
                    't_stat': -23.189668992198083, 'p_value': 1.0},
            'age': {'coefficient': -234.9917187921841, 'standard_error': 16.183832538528538,
                    't_stat': -14.52015264201133, 'p_value': 1.0},
            'educ': {'coefficient': 5774.818729189603, 'standard_error': 142.8572756581845,
                     't_stat': 40.423693526166964, 'p_value': 0.0},
            'white': {'coefficient': 3404.128500195802, 'standard_error': 1156.6740591808912,
                      't_stat': 2.9430317669668025, 'p_value': 0.0016277988173345036},
            'intercept': {'coefficient': 13199.797995938803, 'standard_error': 1893.0142350027074,
                          't_stat': 6.972899491122909, 'p_value': 1.6234517904886598e-12}
        }

        # Compare results for each variable
        for key in expected_results.keys():
            with self.subTest(variable=key):
                # Print expected and actual values for debugging purposes
                print(f"\nVariable: {key}")
                print(f"Expected: {expected_results[key]}")
                print(f"Actual: {reg.results[key]}")

                self.assertAlmostEqual(expected_results[key]['coefficient'], reg.results[key]['coefficient'], places=2,
                                       msg=f"Coefficient mismatch for {key}")
                self.assertAlmostEqual(expected_results[key]['standard_error'], reg.results[key]['standard_error'], places=2,
                                       msg=f"Standard error mismatch for {key}")
                self.assertAlmostEqual(expected_results[key]['t_stat'], reg.results[key]['t_stat'], places=2,
                                       msg=f"T-stat mismatch for {key}")
                self.assertAlmostEqual(expected_results[key]['p_value'], reg.results[key]['p_value'], places=2,
                                       msg=f"P-value mismatch for {key}")
