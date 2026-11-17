"""Interactive live value-solver comparison using the project's core code."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import numpy as np
import pandas as pd
import streamlit as st
from mini1_rl.mdp import GridWorldMRP, solve_value_matrix, solve_value_iterative, solve_value_eigendecomp
from mini1_rl.experiments.masks import mask_assignment_primary

st.title('Finite MDP: live solver comparison')
st.caption('Academic simulation | AI-assisted implementation | Calculated live, not saved report figures')
gamma=st.slider('Discount factor',0.10,0.99,0.90,0.01)
mrp=GridWorldMRP(mask_assignment_primary(),gamma=gamma,terminal_rows=[4])
exact=solve_value_matrix(mrp)
iterative=solve_value_iterative(mrp,tol=1e-10)
spectral=solve_value_eigendecomp(mrp)
error=float(np.max(np.abs(exact-iterative)))
st.metric('Maximum matrix / iterative disagreement',f'{error:.3e}')
st.dataframe(pd.DataFrame({'State':list(map(str,mrp.states)),'Matrix':exact,'Iterative':iterative,'Spectral':spectral}),hide_index=True)
st.bar_chart(pd.DataFrame({'Matrix':exact,'Iterative':iterative}))
st.write('Changing the discount factor recomputes all three solvers on the same transition system. Agreement checks numerical consistency; it does not validate a real-world control policy.')
st.download_button('Download live values',pd.DataFrame({'matrix':exact,'iterative':iterative,'spectral':spectral}).to_csv(index=False),'value-comparison.csv','text/csv')
