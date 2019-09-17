#include <Math.h>
#include <BasicLinearAlgebra.h>
#include "WeightBias.h"
#include "validacion.h"

// matrices temporales
Matrix<1, 15, float> test_;

Matrix<15, 16, float> WL0_;
Matrix<1, 16, float> BL0_;
Matrix<1, 16, float> salida0_;

Matrix<16, 16, float> WL1_;
Matrix<1, 16, float> BL1_;
Matrix<1, 16, float> salida1_;

Matrix<16, 3, float> WL2_;
Matrix<1, 3, float> BL2_;
Matrix<1, 3, float> salida2_;

float salida0[1][16];

float salida1[1][16];

float salida2[1][3];

// valores para testing

float test[1][15] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

int mult_val = 100; // para hacer visibles las predicciones.

int vali_pos = 10;

void setup() {
  Serial.begin(115200);
  startALLvariables();
  for (int i = 0; i < 10; i++) {
    print_val(i);
    NeuralNetwork();
  }

}

void loop() {
}

float sigmoid(float x) {
  return 1.0 / (exp(-x) + 1.0);
}

float relu(float x) {
  if ( x >= 0) {
    return x;
  }
  else {
    return 0;
  }
}


void print_val(int vali_pos) {

  for (int i = 0; i < 15; i++) {
    test[0][i] = validation[vali_pos][i];

  }
  test_ = test;

}

float NeuralNetwork() {
  // cálculo de la capa 0:
  Matrix<1, 16> suma0 = (test_ * WL0_) + BL0_;//f((x*W)+B)
  // función de activación:
  for (int i = 0; i < 16; i++) {
    salida0[0][i] = relu(suma0(0, i));
  }
  salida0_ = salida0;
  // cálculo de la capa 1:
  Matrix<1, 16> suma1 = (salida0_ * WL1_) + BL1_;
  // función de activación:
  for (int i = 0; i < 16; i++) {
    salida1[0][i] = relu(suma1(0, i));
  }
  salida1_ = salida1;
  // cálculo de la capa 2:
  Matrix<1, 3> suma2 = (salida1_ * WL2_) + BL2_;
  // función de activación:
  for (int i = 0; i < 3; i++) {
    salida2[0][i] = sigmoid(suma2(0, i));
  }
  salida2_ = salida2;
  // imprimir resultado
  Serial << "Matrix de salida: " <<  salida2_* mult_val  << '\n';

}

void startALLvariables() {
  WL0_ = WL0;
  BL0_ = BL0;
  WL1_ = WL1;
  BL1_ = BL1;
  WL2_ = WL2;
  BL2_ = BL2;
}


