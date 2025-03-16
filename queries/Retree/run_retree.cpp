#include "duckdb.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

std::string PATH = "/volumn/Retree_exp/queries/";
std::string SQL_PATH = "/volumn/Retree_exp/queries/Retree/workloads/";
std::string MODEL_PATH = "/volumn/Retree_exp/workloads/";

std::string read_file(const std::string &filename)
{
	std::ifstream file(filename);
	if (!file.is_open())
	{
		throw std::runtime_error("Unable to open file: " + filename);
	}
	std::stringstream buffer;
	buffer << file.rdbuf();
	return buffer.str();
}

std::vector<std::string> read_predicates(const std::string &filename)
{
	std::ifstream file(filename);
	if (!file.is_open())
	{
		throw std::runtime_error("Unable to open predicate file: " + filename);
	}
	std::vector<std::string> predicates;
	std::string line;
	while (std::getline(file, line))
	{
		predicates.push_back(line);
	}
	return predicates;
}

std::string replacePlaceholder(std::string str, const std::string &from, const std::string &to)
{
	size_t start_pos = str.find(from);
	if (start_pos != std::string::npos)
	{
		str.replace(start_pos, from.length(), to);
	}
	return str;
}

struct Config
{
	std::string workload = "nyc-taxi-green-dec-2016";
	std::string model = "nyc-taxi-green-dec-2016_d11_l1491_n2981_20250112085333";
	std::string model_type = "rf";
	std::string scale = "1G";
	std::string thread = "4";
	int times = 5;
	int optimization_level = 3;
};

Config parse_args(int argc, char *argv[])
{
	Config config;
	int opt;
	while ((opt = getopt(argc, argv, "t:w:o:m:s:y:")) != -1)
	{
		switch (opt)
		{
		case 'w':
			config.workload = optarg;
			break;
		case 'm':
			config.model = optarg;
			break;
		case 'y':
			config.model_type = optarg;
			break;
		case 's':
			config.scale = optarg;
			break;
		case 't':
			config.thread = optarg;
			break;
		case 'o':
			config.optimization_level = atoi(optarg);
			break;
		default:
			std::cerr << "Usage: " << argv[0]
					  << " [-w workloads] [-m model] [-y model_type] [-s scale] [-t threads] [-o optimization_level]\n";
			exit(EXIT_FAILURE);
		}
	}
	return config;
}

void run(const Config &config)
{
	duckdb::DBConfig db_config;
	db_config.options.allow_unsigned_extensions = true;
	db_config.options.allow_extensions_metadata_mismatch = true;
	duckdb::DuckDB db(nullptr, &db_config);
	duckdb::Connection con(db);

	con.Query("PRAGMA disable_verification;");
	con.Query("set allow_extensions_metadata_mismatch=true;");

	con.Query(read_file(PATH + "load_inference_function.sql"));

	if (config.optimization_level >= 1)
	{
		con.Query(read_file(PATH + "load_convert_rule.sql"));
	}
	if (config.optimization_level >= 2)
	{
		con.Query(read_file(PATH + "load_prune_rule.sql"));
	}
	if (config.optimization_level >= 3)
	{
		con.Query(read_file(PATH + "load_merge_rule.sql"));
	}

	std::string sql_path = SQL_PATH + config.workload + "/";
	std::vector<std::string> predicates = read_predicates(MODEL_PATH + config.workload + "/model/predicates.txt");
	std::ofstream outputfile(sql_path + "output.csv", std::ios::app);

	std::vector<double> records;

	std::string threads = replacePlaceholder("set threads = ?;", "?", config.thread);
	con.Query(threads);

	con.Query(replacePlaceholder(read_file(sql_path + "load_data.sql"), "?", config.scale));
	for (const auto &predicate : predicates)
	{
		std::string sql = read_file(sql_path + "query.sql");
		sql = replacePlaceholder(sql, "?", config.model);
		sql = replacePlaceholder(sql, "?", predicate);

		int count = config.times;

		records.clear();
		while (count--)
		{
			auto start = std::chrono::high_resolution_clock::now();
			con.Query(sql);
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> duration = end - start;
			records.push_back(duration.count());
		}

		int maxminvals =
			*std::max_element(records.begin(), records.end()) + *std::min_element(records.begin(), records.end());
		double sum = std::accumulate(records.begin(), records.end(), 0.0) - maxminvals;
		double average = sum / (records.size() - 2);

		outputfile << config.workload << "," << config.model << "," << predicate << "," << config.scale << ","
				   << config.thread << "," << config.optimization_level << "," << average << "\n";
		std::cout << config.workload << "," << config.model << "," << predicate << "," << config.scale << ","
				  << config.thread << "," << config.optimization_level << "," << average << std::endl;

		if (config.optimization_level == 0)
			break;
	}
	outputfile.close();
}

/* ./build/run_retree -w "nyc-taxi-green-dec-2016" -m "nyc-taxi-green-dec-2016_t100_d10_l843_n1686_20250314065259" -y "rf" -s "1G" -t "4" -o "0" */
int main(int argc, char *argv[])
{
	Config config = parse_args(argc, argv);
	run(config);
}
