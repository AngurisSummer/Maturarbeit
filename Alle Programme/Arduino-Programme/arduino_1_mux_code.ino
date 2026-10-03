#include <Wire.h>
#include "Communication.h"

#define S0 2
#define S1 3
#define S2 4
#define S3 5
#define SIG_PIN A0
#define RED_LED 10
#define GREEN_LED 11
#define SIGNAL_PIN 13  // Added signal pin

static Communication Comm(10); 

const int TRIGGER_THRESHOLD = 400; 
unsigned long extensionStartTime = 0;
bool monitoring = false;
bool hasTriggered = false; 

void selectChannel(int ch) {
  digitalWrite(S0, bitRead(ch, 0));
  digitalWrite(S1, bitRead(ch, 1));
  digitalWrite(S2, bitRead(ch, 2));
  digitalWrite(S3, bitRead(ch, 3));
  delayMicroseconds(50);
}

void setup() {
  Serial.begin(115200);
  pinMode(S0, OUTPUT); pinMode(S1, OUTPUT);
  pinMode(S2, OUTPUT); pinMode(S3, OUTPUT);
  pinMode(SIG_PIN, INPUT_PULLUP);
  
  pinMode(RED_LED, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  pinMode(SIGNAL_PIN, OUTPUT); // Initialize Signal Pin
  
  digitalWrite(SIGNAL_PIN, LOW); // Ensure it starts LOW
  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED, HIGH);
  Serial.println(F("Hub Online. Waiting for Start..."));
}

void loop() {
  int rx = Comm.Received();
  
  if (rx == STRING_MESSAGE) {
    String cmd = Comm.GetStringMessage();
    
    if (cmd == "WAIT_TRIGGER" || cmd == "SET_RED") {
      digitalWrite(RED_LED, HIGH);
      digitalWrite(GREEN_LED, LOW);
      digitalWrite(SIGNAL_PIN, LOW); // Ensure signal is low during wait phase
      extensionStartTime = millis();
      monitoring = true;
      hasTriggered = false; 
      Serial.println(F("\n--- NEW RUN STARTED ---"));
    }
    
    else if (cmd == "SET_GREEN") {
      digitalWrite(RED_LED, LOW);
      digitalWrite(GREEN_LED, HIGH);
      digitalWrite(SIGNAL_PIN, LOW); // Ensure signal is low when finished
      monitoring = false;
      Serial.println(F("\n--- ALL RUNS COMPLETE ---"));
    }
    
    if (cmd == "CHECK_TRIGGER") {
      if (hasTriggered) {
        Comm.Transmit(25, "YES");
        return;
      }

      // 1. Check Sensor 1 (Channel 1)
      selectChannel(3); 
      int sensor1Value = analogRead(SIG_PIN);
      
      // 2. Check Sensor 2 (Channel 2)
      selectChannel(4); 
      int sensor2Value = analogRead(SIG_PIN);

      unsigned long elapsed = millis() - extensionStartTime;

      // 3. Evaluate if BOTH sensors see the object
      if (monitoring && (elapsed > 1000)) {
        if (sensor1Value < TRIGGER_THRESHOLD && sensor2Value < TRIGGER_THRESHOLD) { 
          Serial.println(F("Sensors Tripped! Allowing 1 second of extra travel..."));
          
          delay(1000); // Actuator keeps moving during this second
          
          hasTriggered = true; 
          Comm.Transmit(25, "YES"); // Now tell Arduino 2 to stop and fire
        } else {
          Comm.Transmit(25, "NO");
        }
      } else {
        Comm.Transmit(25, "NO");
      }
    }
    
    else if (cmd == "GET_DATA") {
      // SET SIGNAL HIGH before data processing begins
      digitalWrite(SIGNAL_PIN, HIGH);

      // Part 1: M0 to M4
      String part1 = "";
      for (int i = 0; i < 5; i++) {
        selectChannel(i);
        part1 += String(analogRead(SIG_PIN)) + ",";
      }
      
      // Part 2: M5 to M9
      String part2 = "";
      for (int i = 5; i < 10; i++) {
        selectChannel(i);
        part2 += String(analogRead(SIG_PIN));
        if (i < 9) part2 += ",";
      }

      char buf1[32];
      char buf2[32];
      part1.toCharArray(buf1, 32);
      part2.toCharArray(buf2, 32);

      Comm.Transmit(25, buf1);
      delay(15); 
      Comm.Transmit(25, buf2); 

      // Note: In your logic, Arduino 2 calls this in a rapid loop.
      // D13 will stay HIGH during this entire phase.
      // It will reset to LOW when SET_GREEN or SET_RED is called.
    }
  }
}
