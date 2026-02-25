// ImageMatch.cpp : 定义控制台应用程序的入口点。
//

#include <glog/logging.h>
#include "MultiAlign.h"


void logInit(char** argv)
{
//	string exePath = boost::filesystem::initial_path<boost::filesystem::path>().string();
	FLAGS_log_dir = "Logs";
//	boost::filesystem::create_directory(FLAGS_log_dir);
	FLAGS_alsologtostderr = true;
//	google::InitGoogleLogging(exePath.c_str());
 	google::SetLogDestination(google::GLOG_INFO, (FLAGS_log_dir + "//INFO_").c_str());
 	google::SetLogDestination(google::GLOG_ERROR, (FLAGS_log_dir + "//ERROR_").c_str());

	FLAGS_colorlogtostderr = true;  // Set log color
	FLAGS_logbufsecs = 0;  // Set log output speed(s)                                                                                                                                                           
	FLAGS_max_log_size = 1024;  // Set max log file size
	FLAGS_stop_logging_if_full_disk = true;  // If disk is full
}


int main(int argc, char** argv)
{
	char s[20] = "";
	char s_New[20] = "\0";
	for(int i = 0; i < strlen(s); i++)
	{
	   s_New[i] |= i;
	}
	//s_New = "z{_oeuowhm";


	logInit(argv);

	int step = atoi(argv[1]);
	int method = atoi(argv[2]);

	MultiAlign mss;
	mss.preprocess(0, method);
	/*method: 1:eor,2:lm-only,3:mor,5:gt*/
	/*step: 0:default regist , 1:sample,2:sequence sdf*/
	mss.processRegistHDAndMSSImages(step);
}
