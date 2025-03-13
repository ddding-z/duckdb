// duckdb logical part
#define DUCKDB_EXTENSION_MAIN
#include "duckdb.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/logical_operator.hpp"

using namespace duckdb;

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>

// retree rules part
#include "onnxoptimizer/optimize_c_api/optimize_c_api.h"

#include <iomanip>
#include <regex>

//===--------------------------------------------------------------------===//
// ** ReTree Optimize Rules
//===--------------------------------------------------------------------===//

//** rule2: 合并
class DTMergeExtension : public OptimizerExtension {
public:
	DTMergeExtension() {
		optimize_function = mergeDTRule;
	}

public:
	static bool visitExpression(Expression &expr, int threads_count) {
		if (expr.expression_class == ExpressionClass::BOUND_COMPARISON) {
			auto &comparison_expr = dynamic_cast<BoundComparisonExpression &>(expr);
			if (comparison_expr.left->expression_class == ExpressionClass::BOUND_FUNCTION) {
				auto &func_expr = (BoundFunctionExpression &)*comparison_expr.left;
				if (func_expr.function.name == "predict") {
					auto &first_param = (BoundConstantExpression &)*func_expr.children[0];
					std::string original_model_path = first_param.value.ToString();
					if (original_model_path.find("pruned") != std::string::npos) {

						// std::ofstream outputfile("/volumn/duckdb/examples/embedded-c++/workload/merging_cost.txt",
						//                          std::ios::app);
						// auto start = std::chrono::high_resolution_clock::now();

						// auto opted_model_path =
						//     optimize_on_decision_tree_predicate_merge(original_model_path, threads_count);

						// auto end = std::chrono::high_resolution_clock::now();
						// std::chrono::duration<double, std::milli> duration = end - start;
						// outputfile << "DTMerge time cost (s): "
						//            << duration.count() / 1000 << "\n";
						// outputfile.close();

						auto opted_model_path =
						optimize_on_decision_tree_predicate_merge(original_model_path, threads_count);
						duckdb::Value model_path_value(opted_model_path);
						first_param.value = model_path_value;
						return true;
					}
				}
			}
		}

		bool match = false;
		ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) {
			if (visitExpression(child, threads_count)) {
				match = true;
			}
		});
		return match;
	}

	static bool visitOperator(LogicalOperator &op, int threads_count) {
		for (auto &expr : op.expressions) {
			if (visitExpression(*expr, threads_count)) {
				return true;
			}
		}
		for (auto &child : op.children) {
			if (visitOperator(*child, threads_count)) {
				return true;
			}
		}
		return false;
	}

	static void mergeDTRule(OptimizerExtensionInput &input, duckdb::unique_ptr<LogicalOperator> &plan) {
		int threads_count = input.context.db->NumberOfThreads();
		visitOperator(*plan, threads_count);
	}
};

//===--------------------------------------------------------------------===//
// ** Extension load + setup
//===--------------------------------------------------------------------===//
extern "C" {
DUCKDB_EXTENSION_API void retree_merge_rule_extension_init(duckdb::DatabaseInstance &db) {
	Connection con(db);
	auto &config = DBConfig::GetConfig(db);

	// add a parser extension: 合并
	config.optimizer_extensions.push_back(DTMergeExtension());
	config.AddExtensionOption("merge", "onnx model branch merging", LogicalType::INVALID);
}

DUCKDB_EXTENSION_API const char *retree_merge_rule_extension_version() {
	return DuckDB::LibraryVersion();
}
}
