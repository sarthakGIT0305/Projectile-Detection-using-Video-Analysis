#include "projectile/config.hpp"
#include "projectile/pipeline.hpp"
#include <iostream>
int main(int argc,char**argv){try{projectile::Config c;if(argc>1)c=projectile::load_config(argv[1],c);else{std::cerr<<"Usage: projectile_live <config.ini>\n";return 2;}return projectile::Pipeline(std::move(c)).run();}catch(const std::exception& e){std::cerr<<"projectile_live: "<<e.what()<<"\n";return 1;}}
