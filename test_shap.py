import pandas as pd
import numpy as np
import shap

# let's mock an explainer with lightgbm
import lightgbm as lgb

X = pd.DataFrame(np.random.rand(100, 5), columns=['F1', 'F2', 'F3', 'F4', 'F5'])
y = np.random.randint(0, 2, 100)

clf = lgb.LGBMClassifier()
clf.fit(X, y)

explainer = shap.TreeExplainer(clf)
idx = 0
shap_obj = explainer(X.iloc[[idx]])
print(shap_obj.shape)
if len(shap_obj.shape) == 3:
    sv_patient = shap_obj[0, :, 1]
else:
    sv_patient = shap_obj[0]

import matplotlib.pyplot as plt
shap.waterfall_plot(sv_patient, show=False)
plt.savefig('test_waterfall.png')
print("Waterfall works")
