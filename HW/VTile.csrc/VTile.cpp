// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "VTile.h"
#include "VTile__Syms.h"

//============================================================
// Constructors

VTile::VTile(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new VTile__Syms(contextp(), _vcname__, this)}
    , io_nasti_aw_bits_addr{vlSymsp->TOP.io_nasti_aw_bits_addr}
    , io_nasti_aw_ready{vlSymsp->TOP.io_nasti_aw_ready}
    , io_nasti_w_ready{vlSymsp->TOP.io_nasti_w_ready}
    , io_nasti_w_valid{vlSymsp->TOP.io_nasti_w_valid}
    , io_nasti_w_bits_data{vlSymsp->TOP.io_nasti_w_bits_data}
    , io_nasti_r_bits_last{vlSymsp->TOP.io_nasti_r_bits_last}
    , io_host_fromhost_valid{vlSymsp->TOP.io_host_fromhost_valid}
    , io_host_fromhost_bits{vlSymsp->TOP.io_host_fromhost_bits}
    , io_nasti_b_ready{vlSymsp->TOP.io_nasti_b_ready}
    , io_nasti_r_ready{vlSymsp->TOP.io_nasti_r_ready}
    , io_nasti_r_bits_data{vlSymsp->TOP.io_nasti_r_bits_data}
    , io_nasti_b_valid{vlSymsp->TOP.io_nasti_b_valid}
    , io_nasti_w_bits_last{vlSymsp->TOP.io_nasti_w_bits_last}
    , reset{vlSymsp->TOP.reset}
    , io_nasti_r_valid{vlSymsp->TOP.io_nasti_r_valid}
    , io_nasti_ar_valid{vlSymsp->TOP.io_nasti_ar_valid}
    , io_nasti_ar_bits_addr{vlSymsp->TOP.io_nasti_ar_bits_addr}
    , io_host_tohost{vlSymsp->TOP.io_host_tohost}
    , io_nasti_ar_ready{vlSymsp->TOP.io_nasti_ar_ready}
    , io_nasti_aw_valid{vlSymsp->TOP.io_nasti_aw_valid}
    , clock{vlSymsp->TOP.clock}
    , io_nasti_b_bits_id{vlSymsp->TOP.io_nasti_b_bits_id}
    , io_nasti_b_bits_resp{vlSymsp->TOP.io_nasti_b_bits_resp}
    , io_nasti_r_bits_id{vlSymsp->TOP.io_nasti_r_bits_id}
    , io_nasti_r_bits_resp{vlSymsp->TOP.io_nasti_r_bits_resp}
    , io_nasti_aw_bits_id{vlSymsp->TOP.io_nasti_aw_bits_id}
    , io_nasti_aw_bits_len{vlSymsp->TOP.io_nasti_aw_bits_len}
    , io_nasti_aw_bits_size{vlSymsp->TOP.io_nasti_aw_bits_size}
    , io_nasti_aw_bits_burst{vlSymsp->TOP.io_nasti_aw_bits_burst}
    , io_nasti_aw_bits_lock{vlSymsp->TOP.io_nasti_aw_bits_lock}
    , io_nasti_aw_bits_cache{vlSymsp->TOP.io_nasti_aw_bits_cache}
    , io_nasti_aw_bits_prot{vlSymsp->TOP.io_nasti_aw_bits_prot}
    , io_nasti_aw_bits_qos{vlSymsp->TOP.io_nasti_aw_bits_qos}
    , io_nasti_w_bits_strb{vlSymsp->TOP.io_nasti_w_bits_strb}
    , io_nasti_ar_bits_id{vlSymsp->TOP.io_nasti_ar_bits_id}
    , io_nasti_ar_bits_len{vlSymsp->TOP.io_nasti_ar_bits_len}
    , io_nasti_ar_bits_size{vlSymsp->TOP.io_nasti_ar_bits_size}
    , io_nasti_ar_bits_burst{vlSymsp->TOP.io_nasti_ar_bits_burst}
    , io_nasti_ar_bits_lock{vlSymsp->TOP.io_nasti_ar_bits_lock}
    , io_nasti_ar_bits_cache{vlSymsp->TOP.io_nasti_ar_bits_cache}
    , io_nasti_ar_bits_prot{vlSymsp->TOP.io_nasti_ar_bits_prot}
    , io_nasti_ar_bits_qos{vlSymsp->TOP.io_nasti_ar_bits_qos}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

VTile::VTile(const char* _vcname__)
    : VTile(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

VTile::~VTile() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void VTile___024root___eval_debug_assertions(VTile___024root* vlSelf);
#endif  // VL_DEBUG
void VTile___024root___eval_static(VTile___024root* vlSelf);
void VTile___024root___eval_initial(VTile___024root* vlSelf);
void VTile___024root___eval_settle(VTile___024root* vlSelf);
void VTile___024root___eval(VTile___024root* vlSelf);

void VTile::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate VTile::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    VTile___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        vlSymsp->__Vm_didInit = true;
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        VTile___024root___eval_static(&(vlSymsp->TOP));
        VTile___024root___eval_initial(&(vlSymsp->TOP));
        VTile___024root___eval_settle(&(vlSymsp->TOP));
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    VTile___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool VTile::eventsPending() { return false; }

uint64_t VTile::nextTimeSlot() {
    VL_FATAL_MT(__FILE__, __LINE__, "", "%Error: No delays in the design");
    return 0;
}

//============================================================
// Utilities

const char* VTile::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void VTile___024root___eval_final(VTile___024root* vlSelf);

VL_ATTR_COLD void VTile::final() {
    VTile___024root___eval_final(&(vlSymsp->TOP));
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* VTile::hierName() const { return vlSymsp->name(); }
const char* VTile::modelName() const { return "VTile"; }
unsigned VTile::threads() const { return 16; }
