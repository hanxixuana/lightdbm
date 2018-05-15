#include <iostream>
#include <LightGBM/application.h>

/*!
 *  1. Transformation of Deltas
 *  2. Monotonicity
 *  3. Interaction
 *  4. DART (Note that it may affect the variable importance)
 */

int main(int argc, char** argv) {
  try {
    LightGBM::Application app(argc, argv);
    app.Run();
  }
  catch (const std::exception& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex.what() << std::endl;
    exit(-1);
  }
  catch (const std::string& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex << std::endl;
    exit(-1);
  }
  catch (...) {
    std::cerr << "Unknown Exceptions" << std::endl;
    exit(-1);
  }
}