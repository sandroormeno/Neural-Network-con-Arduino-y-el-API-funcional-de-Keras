#include <Math.h>
#include <BasicLinearAlgebra.h>
#include "WeightBias.h"
#include "validacion.h"

// valores para testing

float test_x[1][5] = { 0.39215686, 0.42745098, 0.4      ,0.35294118, 0.37647059};
float test_y[1][5] = { 0.4745098,  0.48235294, 0.4745098,  0.47058824, 0.4745098};
float test_z[1][5] = { 0.49803922, 0.49411765, 0.49803922, 0.50196078, 0.50196078};

// matrices temporales

Matrix<1, 5, float> test_x_;
Matrix<1, 5, float> test_y_;
Matrix<1, 5, float> test_z_;

Matrix<5, 5, float> WL3_;
Matrix<1, 5, float> BL3_;
Matrix<1, 5, float> salida0_;
Matrix<5, 4, float> WL6_; 
Matrix<1, 4, float> BL6_;
Matrix<1, 4, float> salida1_;

Matrix<5, 5, float> WL4_;
Matrix<1, 5, float> BL4_;
Matrix<1, 5, float> salida2_;
Matrix<5, 4, float> WL7_; 
Matrix<1, 4, float> BL7_;
Matrix<1, 4, float> salida3_;

Matrix<5, 5, float> WL5_;
Matrix<1, 5, float> BL5_;
Matrix<1, 5, float> salida4_;
Matrix<5, 4, float> WL8_; 
Matrix<1, 4, float> BL8_;
Matrix<1, 4, float> salida5_;

Matrix<12, 3, float> WL10_;
Matrix<1, 3, float> BL10_;
Matrix<1, 3, float> salida_;

Matrix<10, 3, int> labels_;

float salida0[1][5];
float salida1[1][4];
float salida2[1][5];
float salida3[1][4];
float salida4[1][5];
float salida5[1][4];
float salida[1][3];

int mult_val = 100; // para hacer visibles las predicciones.

void setup() {
  Serial.begin(115200);
  startALLvariables();
  print_head();
  for (int i = 0; i < 10; i++) {
    print_val(i);
    NeuralNetwork();
    Serial << " \t |  " <<  labels_(i,0) << "  "<< labels_(i,1)<< "  "<< labels_(i,2)<< "  |"<< '\n' ; 
  }
  print_footer();
}

void loop() {
}

void print_val(int vali_pos) {

  for (int i = 0; i < 5; i++) {
    test_x[0][i] = _test_x[vali_pos][i];
    test_y[0][i] = _test_y[vali_pos][i];
    test_z[0][i] = _test_z[vali_pos][i];
  }
  test_x_ = test_x;
  test_y_ = test_y;
  test_z_ = test_z;
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
float NeuralNetwork() { 
  //------ cálculo para datos X ------
  Matrix<1, 5> suma0 = test_x_ * WL3_ + BL3_; //<-- X*W+b
  for (int i = 0; i < 5; i++) {
    salida0[0][i] = relu(suma0(0, i)); //<--función de activación
  }
  salida0_ = salida0;
  Matrix<1, 4> suma1 = salida0_ * WL6_ + BL6_;
  for (int i = 0; i < 4; i++) {
    salida1[0][i] = relu(suma1(0, i)); //<--función de activación
  }
  salida1_ = salida1;
  //------cálculo para datos Y ------
  Matrix<1, 5> suma3 = test_y_ * WL4_ + BL4_;
  for (int i = 0; i < 5; i++) {
    salida2[0][i] = relu(suma3(0, i)); //<--función de activación
  }
  salida2_ = salida2;
  Matrix<1, 4> suma4 = salida2_ * WL7_ + BL7_;
  for (int i = 0; i < 4; i++) {
    salida3[0][i] = relu(suma4(0, i)); //<--función de activación
  }
  salida3_ = salida3;
  //------cálculo para datos Z ------
  Matrix<1, 5> suma5 = test_z_ * WL5_ + BL5_;
  for (int i = 0; i < 5; i++) {
    salida4[0][i] = relu(suma5(0, i)); //<--función de activación
  }
  salida4_ = salida4;
  Matrix<1, 4> suma6 = salida4_ * WL8_ + BL8_;
  for (int i = 0; i < 4; i++) {
    salida5[0][i] = relu(suma6(0, i)); //<--función de activación
  }
  salida5_ = salida5;  
  //-------------- Concatenar  -----------
  Matrix<1,8> conct1 = HorzCat(salida1_,salida3_); // una a la vez
  Matrix<1,12> concatAll = HorzCat(conct1,salida5_);
  Matrix<1, 3> suma7 = concatAll * WL10_ + BL10_;
  Matrix<1, 3> mult = concatAll * WL10_;
  for (int i = 0; i < 3; i++) {
    salida[0][i] = sigmoid(suma7(0, i)); //<--función de activación
  }
  salida_ = salida;
  //-------------- Salida Final  -----------
  Serial << "|   " <<  salida_(0,0)*mult_val <<"  "<<salida_(0,1)*mult_val <<"  "<< salida_(0,2)*mult_val ;  
}

void startALLvariables() {

  WL3_ = WL3;
  BL3_ = BL3;
  WL6_ = WL6;
  BL6_ = BL6;

  WL4_ = WL4;
  BL4_ = BL4;
  WL7_ = WL7;
  BL7_ = BL7;

  WL5_ = WL5;
  BL5_ = BL5; 
  WL8_ = WL8; 
  BL8_ = BL8;

  WL10_ = WL10;
  BL10_ = BL10;
  labels_ = labels;
}


void print_head(){

  Serial.println("|------------------------|-----------|");
  Serial.println("|      predic labels     |   real    |");
  Serial.println("|------------------------|-----------|");
}
void print_footer(){
  Serial.println("|------------------------|-----------|");
}

