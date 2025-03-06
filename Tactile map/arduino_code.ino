#include <Servo.h>

Servo actuator;  // Servo or solenoid actuator

void setup() {
  Serial.begin(115200);
  actuator.attach(9); // Pin for actuator control
  actuator.write(0);  // Default: retracted
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    Serial.println("Received: " + command);

    if (command.startsWith("M3")) { 
      actuator.write(90); // Raise dot
    } 
    else if (command.startsWith("M5")) { 
      actuator.write(0); // Lower dot
    }
  }
}
