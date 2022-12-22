
#include "Matrix.cuh"

using T = unsigned long;
float Matrix<T>::last_multiplication_time = 0.0f;

/*
	filename[0] Matrix A file
	filename[1] Matrix B file
	filename[2] Output file
*/
bool check_python(const std::string filename[3])
{
	const char* verificaton_result = "verification.txt";

	std::cout << "Checking results by Python's NumPy";
	system(("python verificatior.py " + filename[0] + ' ' + filename[1] + ' ' + filename[2] + " > " + verificaton_result).c_str());

	std::ifstream file;
	file.exceptions(std::ifstream::badbit);
	file.open(verificaton_result);

	std::string buffer = "False";
	getline(file, buffer);
	file.close();
	remove(verificaton_result);

	return buffer != "False";
}

int main(int argc, char** argv)
{
	system("title Parallel Programming [Lab №4]");
	if (argc != 1 && argc != 4 && (argc == 2 && strcmp(argv[1], "--help") == 0))
	{
		std::cout << "Locate paths to matrix files in arguments, to output file and specify number of threads\n\n"
			<< "EXAMPLE:\n"
			<< "    .../PP_3.exe <matrix_1_path> <matrix_2_path> <output_path>\n\n";
		exit(EXIT_SUCCESS);
	}

	/*
	[0] Matrix A file
	[1] Matrix B file
	[2] Output file
	*/
	std::string filename[3];

	if (argc == 4)
	{
		for (int i = 1; i < argc; i++)
			filename[i] = argv[i];
	}
	else
	{
		std::cout << "Locate path to matrix A: ";
		std::cin >> filename[0];

		std::cout << "Locate path to matrix B: ";
		std::cin >> filename[1];

		std::cout << "Locate path to output file: ";
		std::cin >> filename[2];
		std::cout << '\n';
	}

	try
	{

		std::cout << "Reading matrix A";
		Matrix<T> a(filename[0]);

		std::cout << "\rReading matrix B";
		Matrix<T> b(filename[1]);

		std::cout << "\rPerforming C = A * B";
		Matrix<T> c = a * b;

		std::cout << "\rWriting matrix C to file [" << filename[2] << "]";
		c.write_file(filename[2]);
		std::cout << '\r' << std::string(filename[2].size() + 40, ' ') << '\r';

		if (check_python(filename))
		{
			std::cout << "\rAdding multiplication results in [" << filename[2] << "]...";
			c.write_multiplication_result(filename[2]);
			c.write_multiplication_result("res/cuda_res.txt");

			std::cout << '\r' << std::string(filename[2].size() + 40, ' ');

			std::cout << "\rMatrix multiplication was done correctly\n"
				"See results in [" << filename[2] << "]";
		}
		else
			std::cout << "\rMatrix multiplication wasn't done correctly";
	}
	catch (std::exception const& ex)
	{
		std::cout << "\n\n[!] ERROR [!]\n" << ex.what() << "\n\n";
		exit(EXIT_FAILURE);
	}

	std::cout << "\n\n";
	return 0;
}
