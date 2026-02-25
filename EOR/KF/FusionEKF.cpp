#include "FusionEKF.h"
//#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include "HNMath/TransRotation.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


MatrixXd CalculateJacobian(const VectorXd& x_state) {

	MatrixXd Hj(3, 4);

	// Unroll state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	// Pre-compute some term which recur in the Jacobian
	float c1 = px * px + py * py;
	float c2 = sqrt(c1);
	float c3 = c1 * c2;

	// Sanity check to avoid division by zero
	if (std::abs(c1) < 0.0001) {
		std::cout << "Error in CalculateJacobian. Division by zero." << std::endl;
		return Hj;
	}

	// Actually compute Jacobian matrix
	Hj << (px / c2), (py / c2), 0, 0,
		-(py / c1), (px / c1), 0, 0,
		py* (vx * py - vy * px) / c3, px* (vy * px - vx * py) / c3, px / c2, py / c2;

	return Hj;

}

/* Constructor. */
FusionEKF::FusionEKF(const vector<double>& pose) {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // Initialize measurement covariance matrix - M_ini
  R_ini_ = MatrixXd(7, 7);
  R_ini_ << 0.1, 0, 0, 0, 0, 0,0,
				  0, 0.1, 0, 0, 0, 0, 0,
				  0, 0, 0.1, 0, 0, 0, 0,
				  0, 0, 0, 0.1, 0, 0, 0,
				  0, 0, 0, 0, 0.1, 0, 0,
				  0, 0, 0, 0, 0, 0.1, 0,
				  0, 0, 0, 0, 0, 0, 0.1;

  //measurement covariance matrix - M_MAP
  //R是观测噪声
  R_map_ = MatrixXd(7, 7);
  R_map_ << 0.02, 0, 0, 0, 0, 0, 0,
					  0, 0.02, 0, 0, 0, 0, 0,
					  0, 0, 0.02, 0, 0, 0, 0,
					  0, 0, 0, 0.02, 0, 0, 0,
					  0, 0, 0, 0, 0.02, 0, 0,
					  0, 0, 0, 0, 0, 0.02, 0, 
					  0, 0, 0, 0, 0, 0, 0.02;

  // ini - measurement matrix
  H_ini_ = MatrixXd::Identity(7, 7);
  //H_ini_	<<	1, 0, 0, 0,
		//		0, 1, 0, 0;
  H_map_ = MatrixXd::Identity(7, 7);

  // map - jacobian matrix
  Hj_ = MatrixXd::Identity(7, 7);
  //Hj_		<<	0, 0, 0, 0,
		//		0, 0, 0, 0,
		//		0, 0, 0, 0;

  // Initialize state covariance matrix P
  ekf_.P_ = MatrixXd::Identity(7, 7);
  //ekf_.P_	<<	1,	0,	0,	 0,
		//		0,	1,	0,	 0,
		//		0,	0, 1000, 0,
		//		0,	0, 0,	1000;

  // Initial transition matrix F_
  ekf_.F_ = MatrixXd::Identity(7, 7);
  //ekf_.F_ <<	1, 0, 1, 0,
		//		0, 1, 0, 1,
		//		0, 0, 1, 0,
		//		0, 0, 0, 1;

  //过程噪声 Initialize process noise covariance matrix
  ekf_.Q_ = MatrixXd::Zero(7, 7);
  //ekf_.Q_ <<	0, 0, 0, 0,
		//		0, 0, 0, 0,
		//		0, 0, 0, 0,
		//		0, 0, 0, 0;

  // Initialize ekf state
  ekf_.x_ = VectorXd::Zero(7);
  //ekf_.x_ << 1, 1, 1, 1;


  /*****************************************************************************
 *  Initialization
 ****************************************************************************/
  if (!is_initialized_)
  {
	  // Initialize state
	  vector<double> pose_q;
	  CalibSpace::pose6DoFToQuaternion(pose, pose_q);
	  ekf_.x_ = VectorXd::Zero(7);
	  for (int i = 0; i < 7; i++)
	  {
		  ekf_.x_(i) = pose_q[i];
	  }
	  is_initialized_ = true;
  }
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

vector<double> FusionEKF::ProcessMeasurement(const CamPose&cp, bool is_in_intersection) {
  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  // Update state transition matrix F (according to elapsed time dt)
  //ekf_.Q_	<<	dt_4 / 4 * noise_ax,	0,						dt_3 / 2 * noise_ax,	0,
		//		0,						dt_4 / 4 * noise_ay,	0,						dt_3 / 2 * noise_ay,
		//		dt_3 / 2 * noise_ax,	0,						dt_2 * noise_ax,		0,
		//		0,						dt_3 / 2 * noise_ay,	0,						dt_2 * noise_ay;

  
	ekf_.Predict();

	double scale = 1.0;
	if (cp.regist_probability > 0.4)
	{
	//	scale = 0.1;
	}

	if (is_in_intersection)
	{
	//	scale = 100.0;
	}
  /*****************************************************************************
   *  Update
   ****************************************************************************/
  ekf_.R_ = R_map_ * scale;
  ekf_.H_ = H_map_;

  //测量值
  VectorXd z_q = VectorXd::Zero(7);
  vector<double> pose_q;
  CalibSpace::pose6DoFToQuaternion(cp.camPose, pose_q);
  for (int i = 0; i < 7; i++)
  {
	  z_q(i) = pose_q[i];
  }
  ekf_.Update(z_q);
  
  VectorXd x_q = ekf_.x_;
  vector<double> x_(6);
  CalibSpace::QuaternionTopose6DoF(x_q, x_);
  return x_;
}
