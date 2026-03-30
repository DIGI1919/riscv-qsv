// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef VERILATED_VTILE__SYMS_H_
#define VERILATED_VTILE__SYMS_H_  // guard

#include "verilated.h"

// INCLUDE MODEL CLASS

#include "VTile.h"

// INCLUDE MODULE CLASSES
#include "VTile___024root.h"

// SYMS CLASS (contains all model state)
class VTile__Syms final : public VerilatedSyms {
  public:
    // INTERNAL STATE
    VTile* const __Vm_modelp;
    bool __Vm_didInit = false;

    // MULTI-THREADING
    VlThreadPool* const __Vm_threadPoolp;
    bool __Vm_even_cycle__ico = false;
    bool __Vm_even_cycle__act = false;
    bool __Vm_even_cycle__nba = false;

    // MODULE INSTANCE STATE
    VTile___024root                TOP;

    // CONSTRUCTORS
    VTile__Syms(VerilatedContext* contextp, const char* namep, VTile* modelp);
    ~VTile__Syms();

    // METHODS
    const char* name() { return TOP.name(); }
} VL_ATTR_ALIGNED(VL_CACHE_LINE_BYTES);

#endif  // guard
