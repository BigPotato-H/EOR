#ifndef FusionEKF_H_
#define FusionEKF_H_

#include "..\DataSet.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "kalman_filter.h"
//#include "tools.h"

class FusionEKF {

public:

  /* Constructor. */
  FusionEKF(const vector<double>& pose);

  /* Destructor. */
  virtual ~FusionEKF();

  /* Run the whole flow of the Kalman Filter from here. */
  vector<double> ProcessMeasurement(const CamPose &measurement_pack,
	  bool is_in_intersection);

  /* Kalman Filter update and prediction math lives in here. */
  KalmanFilter ekf_;

private:

  // check whether the tracking toolbox was initialized or not (first measurement)
  bool is_initialized_;

  // previous timestamp
  long long previous_timestamp_;

  // tool object used to compute Jacobian and RMSE
  //Tools tools;

  Eigen::MatrixXd R_ini_;
  Eigen::MatrixXd R_map_;
  Eigen::MatrixXd H_ini_;
  Eigen::MatrixXd H_map_;
  Eigen::MatrixXd Hj_;

};

#endif /* FusionEKF_H_ */
