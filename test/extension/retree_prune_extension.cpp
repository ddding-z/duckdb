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

//** rule1: 剪枝
class ReTreePruneExtension : public OptimizerExtension {
public:
	ReTreePruneExtension() {
		optimize_function = retreePruneRule;
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
					if (comparison_expr.right->type == ExpressionType::VALUE_CONSTANT) {
						auto &constant_expr = (BoundConstantExpression &)*comparison_expr.right;
						auto predicate = constant_expr.value.GetValue<float_t>();
						auto comparison_operator = comparison_expr.type;
						uint8_t comparison_operator_ = 0;
						switch (comparison_operator) {
						case ExpressionType::COMPARE_EQUAL:
							comparison_operator_ = 0;
							break;
						case ExpressionType::COMPARE_LESSTHAN:
							comparison_operator_ = 1;
							break;
						case ExpressionType::COMPARE_LESSTHANOREQUALTO:
							comparison_operator_ = 2;
							break;
						case ExpressionType::COMPARE_GREATERTHAN:
							comparison_operator_ = 3;
							break;
						case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
							comparison_operator_ = 4;
							break;
						default:
							comparison_operator_ = 9;
							break;
						}
						if (comparison_operator_ == 9) {
							return true;
						}
						// timer
						// std::ofstream outputfile("/volumn/Retree_exp/prune.csv",
						//                          std::ios::app);
						// auto start = std::chrono::high_resolution_clock::now();
						// auto opted_model_path = optimize_on_decision_tree_predicate_prune(
						//     original_model_path, comparison_operator_, predicate, threads_count);
						// auto end = std::chrono::high_resolution_clock::now();
						// std::chrono::duration<double, std::milli> duration = end - start;
						// outputfile << opted_model_path << "," << duration.count() << "\n";
						// outputfile.close();

						// original_model_path: after convert model path 
						auto opted_model_path = optimize_on_decision_tree_predicate_opt_level_0(
						    original_model_path, comparison_operator_, predicate, threads_count);
						// auto opted_model_path = optimize_on_decision_tree_predicate_prune(
						//     original_model_path, comparison_operator_, predicate, threads_count);
						if (opted_model_path != original_model_path) {
							comparison_expr.type = ExpressionType::COMPARE_GREATERTHANOREQUALTO;
							// std::regex re("_reg(?=\\.onnx$)");
							// original_model_path = regex_replace(original_model_path, re, "");
							// float predicate_updated = 1.0f / get_decision_tree_labels_size(original_model_path);
							// duckdb::Value value(predicate_updated);
							// set comparison_expr: result >= 1/2
							duckdb::Value value(0.5f);
							auto new_constant_expr = std::make_unique<duckdb::BoundConstantExpression>(value);
							comparison_expr.right = std::move(new_constant_expr);
							duckdb::Value model_path_value(opted_model_path);
							first_param.value = model_path_value;
						}
					}
					return true;
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

	static void retreePruneRule(OptimizerExtensionInput &input, duckdb::unique_ptr<LogicalOperator> &plan) {
		int threads_count = input.context.db->NumberOfThreads();
		visitOperator(*plan, threads_count);
	}
};

//===--------------------------------------------------------------------===//
// ** Extension load + setup
//===--------------------------------------------------------------------===//
extern "C" {
DUCKDB_EXTENSION_API void retree_prune_extension_init(duckdb::DatabaseInstance &db) {
	Connection con(db);
	auto &config = DBConfig::GetConfig(db);
	
    // add a parser extension: Retree rules: convert, prune, merge
	config.optimizer_extensions.push_back(ReTreePruneExtension());
	config.AddExtensionOption("ReTree Prune Optimization", "convert, prune", LogicalType::INVALID);

}

DUCKDB_EXTENSION_API const char *retree_prune_extension_version() {
	return DuckDB::LibraryVersion();
}
}
