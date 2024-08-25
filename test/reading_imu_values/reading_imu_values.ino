/*
Test performed 08/24/24 to read in imu values

reading in imu values every dt period & printing it to screen

*/

#include <BasicLinearAlgebra.h>
#include <Arduino_LSM9DS1.h>

using namespace BLA;

#define DELTA_T 25 // ms

// creating imu sensor vectors
BLA::Matrix<3, 1, float> acceleration;
BLA::Matrix<3, 1, float> angular_velocity;

void setup() {

  // To verify output
  Serial.begin(9600);
  while (!Serial);
  Serial.println("Started");

  // initializing IMU
  if (!IMU.begin()) 
  {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // Sample rate is 140 Hz or about 7 ms
  IMU.accelerationSampleRate();
  IMU.gyroscopeSampleRate();

  // zeroing out vectors
  acceleration.Fill(0);
  angular_velocity.Fill(0);

}

void loop() {
  
  if( IMU.accelerationAvailable() )
  {
    IMU.readAcceleration(acceleration(0), acceleration(1), acceleration(2));

    Serial.print("acceleration: ");
    Serial.println(acceleration);
  }

  if( IMU.gyroscopeAvailable() )
  {
    IMU.readGyroscope(angular_velocity(0), angular_velocity(1), angular_velocity(2));

    Serial.print("angular_velocity: ");
    Serial.println(angular_velocity);
  }

  delay(DELTA_T);

}
