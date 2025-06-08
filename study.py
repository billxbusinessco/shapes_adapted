
from study_library import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

analysis_class = analysis(100, 4, False)
#lol = analysis_class.generate_samples()
#analysis_class.generate_data_indicators()
analysis_class.n_shapes = 50

score_array, df1_array = analysis_class.run_study_mindiff_score(number_of_samples=20)
