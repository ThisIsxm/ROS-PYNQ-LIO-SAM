set Project     hls_build.prj
set Solution    solution1
set Device      "xck26-sfvc784-2LV-c"
set Flow        "vivado"
set Clock       5

open_project $Project -reset

set_top fpga_feature_extract_hw

add_files fpga_feature_extract_hw.cpp -cflags -I.
add_files registration_fpga.h -cflags -I.
add_files registration_fpga.cpp -cflags -I.

# default vivado flow
open_solution -reset $Solution      
set_part $Device
create_clock -period $Clock -name default

# csim_design
csynth_design
# cosim_design
export_design

exit