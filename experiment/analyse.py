from analyse import read_result
from analyse import compare_result
from analyse import calc_pearson_correlation
import calc_model_result

# human = read_result.read_result("result/result.txt", "mean")
# top, bottom = calc_model_result.calc_model_result()
# compare_result.compare_result(human, top, bottom, 4)

human = read_result.read_result("result/result.txt", "mean")
top, bottom = calc_model_result.calc_model_result()
calc_pearson_correlation.calc_pearson_correlation(human, top, bottom)

