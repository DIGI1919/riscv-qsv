// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VTile.h for the primary calling header

#include "verilated.h"

#include "VTile___024root.h"

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__596(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__596\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_7 
        = (VL_GTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_5), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_4))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_5)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_4));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__597(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__597\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_7 
        = (VL_LTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_5), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_4))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_5)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_4));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__598(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__598\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulX_T_3 
        = (0x3ffffffU & ((2U == (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm))
                          ? ((0x3fc0000U & ((- (IData)(
                                                       (1U 
                                                        & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulX4b 
                                                           >> 0x11U)))) 
                                            << 0x12U)) 
                             | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulX4b)
                          : ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                              ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_0 
                                 + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_1 
                                    + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_2 
                                       + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_3)))
                              : (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_0 
                                 + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_1 
                                    + (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_2 
                                       + vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_3))))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__599(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__599\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_9 
        = (VL_GTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_7), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_5))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_7)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_5));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__600(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__600\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_9 
        = (VL_LTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_7), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_5))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_7)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_5));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__601(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__601\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___mul8x8_io_out 
        = ((4U == (IData)(vlSelf->__VdfgTmp_h6297395b__0))
            ? (((- (IData)((1U & (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulX_T_3 
                                  >> 0x19U)))) << 0x1aU) 
               | vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___mulX_T_3)
            : ((2U == (IData)(vlSelf->__VdfgTmp_h6297395b__0))
                ? ((2U == (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm))
                    ? ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_36)
                          ? ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                              ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_7
                              : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_7)
                          : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3588fd5f__0)) 
                        << 0x1cU) | ((0xf000000U & 
                                      (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_40)
                                         ? ((0x80U 
                                             & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                             ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_6
                                             : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_6)
                                         : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82d3f__0)) 
                                       << 0x18U)) | 
                                     ((0xf00000U & 
                                       (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_34)
                                          ? ((0x80U 
                                              & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                              ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_5
                                              : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_5)
                                          : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35b14015__0)) 
                                        << 0x14U)) 
                                      | ((0xf0000U 
                                          & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_39)
                                               ? ((0x80U 
                                                   & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                   ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_4
                                                   : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_4)
                                               : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15236__0)) 
                                             << 0x10U)) 
                                         | ((0xf000U 
                                             & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_32)
                                                  ? 
                                                 ((0x80U 
                                                   & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                   ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_3
                                                   : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_3)
                                                  : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3510e805__0)) 
                                                << 0xcU)) 
                                            | ((0xf00U 
                                                & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_38)
                                                     ? 
                                                    ((0x80U 
                                                      & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                      ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_2
                                                      : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_2)
                                                     : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77273__0)) 
                                                   << 8U)) 
                                               | ((0xf0U 
                                                   & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_30)
                                                        ? 
                                                       ((0x80U 
                                                         & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                         ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_1
                                                         : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_1)
                                                        : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35f007dc__0)) 
                                                      << 4U)) 
                                                  | (0xfU 
                                                     & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_37)
                                                         ? 
                                                        ((0x80U 
                                                          & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                          ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_0
                                                          : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_0)
                                                         : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ed03__0))))))))))
                    : ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_40)
                          ? ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                              ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_3
                              : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_3)
                          : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82a30__0)) 
                        << 0x18U) | ((0xff0000U & (
                                                   ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_39)
                                                     ? 
                                                    ((0x80U 
                                                      & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                      ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_2
                                                      : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_2)
                                                     : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15333__0)) 
                                                   << 0x10U)) 
                                     | ((0xff00U & 
                                         (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_38)
                                            ? ((0x80U 
                                                & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_1
                                                : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_1)
                                            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77d60__0)) 
                                          << 8U)) | 
                                        (0xffU & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_37)
                                                   ? 
                                                  ((0x80U 
                                                    & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                    ? vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_0
                                                    : vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_0)
                                                   : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ec00__0)))))))
                : ((1U == (IData)(vlSelf->__VdfgTmp_h6297395b__0))
                    ? ((2U == (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm))
                        ? ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_36)
                              ? ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                  ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_7 
                                     >> 4U) : (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_7 
                                               >> 4U))
                              : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3588fd5f__0)) 
                            << 0x1cU) | ((0xf000000U 
                                          & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_40)
                                               ? ((0x80U 
                                                   & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                   ? 
                                                  (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_6 
                                                   >> 4U)
                                                   : 
                                                  (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_6 
                                                   >> 4U))
                                               : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82d3f__0)) 
                                             << 0x18U)) 
                                         | ((0xf00000U 
                                             & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_34)
                                                  ? 
                                                 ((0x80U 
                                                   & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                   ? 
                                                  (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_5 
                                                   >> 4U)
                                                   : 
                                                  (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_5 
                                                   >> 4U))
                                                  : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35b14015__0)) 
                                                << 0x14U)) 
                                            | ((0xf0000U 
                                                & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_39)
                                                     ? 
                                                    ((0x80U 
                                                      & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                      ? 
                                                     (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_4 
                                                      >> 4U)
                                                      : 
                                                     (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_4 
                                                      >> 4U))
                                                     : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15236__0)) 
                                                   << 0x10U)) 
                                               | ((0xf000U 
                                                   & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_32)
                                                        ? 
                                                       ((0x80U 
                                                         & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                         ? 
                                                        (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_3 
                                                         >> 4U)
                                                         : 
                                                        (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_3 
                                                         >> 4U))
                                                        : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3510e805__0)) 
                                                      << 0xcU)) 
                                                  | ((0xf00U 
                                                      & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_38)
                                                           ? 
                                                          ((0x80U 
                                                            & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                            ? 
                                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_2 
                                                            >> 4U)
                                                            : 
                                                           (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_2 
                                                            >> 4U))
                                                           : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77273__0)) 
                                                         << 8U)) 
                                                     | ((0xf0U 
                                                         & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_30)
                                                              ? 
                                                             ((0x80U 
                                                               & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                               ? 
                                                              (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_1 
                                                               >> 4U)
                                                               : 
                                                              (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_1 
                                                               >> 4U))
                                                              : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35f007dc__0)) 
                                                            << 4U)) 
                                                        | (0xfU 
                                                           & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_37)
                                                               ? 
                                                              ((0x80U 
                                                                & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                                ? 
                                                               (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS4out_0 
                                                                >> 4U)
                                                                : 
                                                               (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU4out_0 
                                                                >> 4U))
                                                               : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ed03__0))))))))))
                        : ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_40)
                              ? ((0x80U & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                  ? (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_3 
                                     >> 8U) : (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_3 
                                               >> 8U))
                              : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35d82a30__0)) 
                            << 0x18U) | ((0xff0000U 
                                          & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_39)
                                               ? ((0x80U 
                                                   & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                   ? 
                                                  (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_2 
                                                   >> 8U)
                                                   : 
                                                  (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_2 
                                                   >> 8U))
                                               : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35e15333__0)) 
                                             << 0x10U)) 
                                         | ((0xff00U 
                                             & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_38)
                                                  ? 
                                                 ((0x80U 
                                                   & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                   ? 
                                                  (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_1 
                                                   >> 8U)
                                                   : 
                                                  (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_1 
                                                   >> 8U))
                                                  : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h35c77d60__0)) 
                                                << 8U)) 
                                            | (0xffU 
                                               & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT___GEN_37)
                                                   ? 
                                                  ((0x80U 
                                                    & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm)
                                                    ? 
                                                   (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulS8out_0 
                                                    >> 8U)
                                                    : 
                                                   (vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT__mulU8out_0 
                                                    >> 8U))
                                                   : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mul8x8__DOT____VdfgTmp_h3527ec00__0)))))))
                    : ((3U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                        ? vlSelf->Tile__DOT__core__DOT__dpath__DOT___pe_ctrl_io_rs_alu_rs1
                        : 0U))));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__602(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__602\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_11 
        = (VL_GTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_9), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_6))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_9)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_6));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__603(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__603\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_11 
        = (VL_LTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_9), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_6))
            ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_9)
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_6));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__604(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__604\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4 
        = ((2U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8mm8_pe_8mm8_op))
            ? (VL_LTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_11), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_7))
                ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___min4_T_11)
                : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_7))
            : ((1U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8mm8_pe_8mm8_op))
                ? (VL_GTS_III(5, (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_11), (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_7))
                    ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___max4_T_11)
                    : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_7))
                : 0U));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__605(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__605\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT___out_T_3 
        = ((2U == (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm))
            ? ((0x1e0U & ((- (IData)((1U & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4) 
                                            >> 4U)))) 
                          << 5U)) | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4))
            : (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out8));
}

VL_INLINE_OPT void VTile___024root___nba_sequent__TOP__606(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___nba_sequent__TOP__606\n"); );
    // Body
    vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___mm8x8_io_qvm_out 
        = ((0U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_io_8mm8_pe_8mm8_op))
            ? 0U : ((2U == (0xfU & vlSelf->Tile__DOT__core__DOT__dpath__DOT__csr__DOT__qnorm))
                     ? ((0xc000U & ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4) 
                                      == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_7))
                                      ? (2U | ((4U 
                                                == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                                                ? ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                   >> 0xeU)
                                                : 0U))
                                      : ((4U == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                         & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                            >> 0xeU))) 
                                    << 0xeU)) | ((0x3000U 
                                                  & ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4) 
                                                       == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_6))
                                                       ? 
                                                      (2U 
                                                       | ((4U 
                                                           == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                                                           ? 
                                                          ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                           >> 0xcU)
                                                           : 0U))
                                                       : 
                                                      ((4U 
                                                        == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                       & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                          >> 0xcU))) 
                                                     << 0xcU)) 
                                                 | ((0xc00U 
                                                     & ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4) 
                                                          == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_5))
                                                          ? 
                                                         (2U 
                                                          | ((4U 
                                                              == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                                                              ? 
                                                             ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                              >> 0xaU)
                                                              : 0U))
                                                          : 
                                                         ((4U 
                                                           == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                          & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                             >> 0xaU))) 
                                                        << 0xaU)) 
                                                    | ((0x300U 
                                                        & ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4) 
                                                             == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_4))
                                                             ? 
                                                            (2U 
                                                             | ((4U 
                                                                 == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                                                                 ? 
                                                                ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                                 >> 8U)
                                                                 : 0U))
                                                             : 
                                                            ((4U 
                                                              == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                             & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                                >> 8U))) 
                                                           << 8U)) 
                                                       | ((0xc0U 
                                                           & ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4) 
                                                                == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_3))
                                                                ? 
                                                               (2U 
                                                                | ((4U 
                                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                                                                    ? 
                                                                   ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                                    >> 6U)
                                                                    : 0U))
                                                                : 
                                                               ((4U 
                                                                 == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                                & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                                   >> 6U))) 
                                                              << 6U)) 
                                                          | ((0x30U 
                                                              & ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4) 
                                                                   == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_2))
                                                                   ? 
                                                                  (2U 
                                                                   | ((4U 
                                                                       == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                                                                       ? 
                                                                      ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                                       >> 4U)
                                                                       : 0U))
                                                                   : 
                                                                  ((4U 
                                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                                   & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                                      >> 4U))) 
                                                                 << 4U)) 
                                                             | ((0xcU 
                                                                 & ((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4) 
                                                                      == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_1))
                                                                      ? 
                                                                     (2U 
                                                                      | ((4U 
                                                                          == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                                                                          ? 
                                                                         ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                                          >> 2U)
                                                                          : 0U))
                                                                      : 
                                                                     ((4U 
                                                                       == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                                      & ((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in) 
                                                                         >> 2U))) 
                                                                    << 2U)) 
                                                                | (3U 
                                                                   & (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out4) 
                                                                       == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs4_0))
                                                                       ? 
                                                                      (2U 
                                                                       | ((4U 
                                                                           == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel))
                                                                           ? (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in)
                                                                           : 0U))
                                                                       : 
                                                                      ((4U 
                                                                        == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT___pe_ctrl_io_pe_sel)) 
                                                                       & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT____Vcellinp__pe_ctrl__io_pe_qvm_in)))))))))))
                     : (((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out8) 
                           == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_3))
                           ? (2U | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94fb19e9__0))
                           : (0xdU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94fb19e9__0))) 
                         << 0xcU) | (((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out8) 
                                        == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_2))
                                        ? (2U | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94aae4d7__0))
                                        : (0xdU & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94aae4d7__0))) 
                                      << 8U) | (((((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out8) 
                                                   == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_1))
                                                   ? 
                                                  (2U 
                                                   | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94dbb1b8__0))
                                                   : 
                                                  (0xdU 
                                                   & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h94dbb1b8__0))) 
                                                 << 4U) 
                                                | (((IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__out8) 
                                                    == (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT__rs8_0))
                                                    ? 
                                                   (2U 
                                                    | (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h940d9fbf__0))
                                                    : 
                                                   (0xdU 
                                                    & (IData)(vlSelf->Tile__DOT__core__DOT__dpath__DOT__pe_unit__DOT__mm8x8__DOT____VdfgTmp_h940d9fbf__0))))))));
}

void VTile___024root___eval_triggers__ico(VTile___024root* vlSelf);
#ifdef VL_DEBUG
VL_ATTR_COLD void VTile___024root___dump_triggers__ico(VTile___024root* vlSelf);
#endif  // VL_DEBUG
void VTile___024root___eval_ico(VTile___024root* vlSelf);
void VTile___024root___eval_triggers__act(VTile___024root* vlSelf);
#ifdef VL_DEBUG
VL_ATTR_COLD void VTile___024root___dump_triggers__act(VTile___024root* vlSelf);
#endif  // VL_DEBUG
void VTile___024root___eval_act(VTile___024root* vlSelf);
#ifdef VL_DEBUG
VL_ATTR_COLD void VTile___024root___dump_triggers__nba(VTile___024root* vlSelf);
#endif  // VL_DEBUG
void VTile___024root___eval_nba(VTile___024root* vlSelf);

void VTile___024root___eval(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___eval\n"); );
    // Init
    CData/*0:0*/ __VicoContinue;
    VlTriggerVec<1> __VpreTriggered;
    IData/*31:0*/ __VnbaIterCount;
    CData/*0:0*/ __VnbaContinue;
    // Body
    vlSelf->__VicoIterCount = 0U;
    __VicoContinue = 1U;
    while (__VicoContinue) {
        __VicoContinue = 0U;
        VTile___024root___eval_triggers__ico(vlSelf);
        if (vlSelf->__VicoTriggered.any()) {
            __VicoContinue = 1U;
            if (VL_UNLIKELY((0x64U < vlSelf->__VicoIterCount))) {
#ifdef VL_DEBUG
                VTile___024root___dump_triggers__ico(vlSelf);
#endif
                VL_FATAL_MT("/home/digi/code/c/riscv-qv/Tile.sv", 3491, "", "Input combinational region did not converge.");
            }
            vlSelf->__VicoIterCount = ((IData)(1U) 
                                       + vlSelf->__VicoIterCount);
            VTile___024root___eval_ico(vlSelf);
        }
    }
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        __VnbaContinue = 0U;
        vlSelf->__VnbaTriggered.clear();
        vlSelf->__VactIterCount = 0U;
        vlSelf->__VactContinue = 1U;
        while (vlSelf->__VactContinue) {
            vlSelf->__VactContinue = 0U;
            VTile___024root___eval_triggers__act(vlSelf);
            if (vlSelf->__VactTriggered.any()) {
                vlSelf->__VactContinue = 1U;
                if (VL_UNLIKELY((0x64U < vlSelf->__VactIterCount))) {
#ifdef VL_DEBUG
                    VTile___024root___dump_triggers__act(vlSelf);
#endif
                    VL_FATAL_MT("/home/digi/code/c/riscv-qv/Tile.sv", 3491, "", "Active region did not converge.");
                }
                vlSelf->__VactIterCount = ((IData)(1U) 
                                           + vlSelf->__VactIterCount);
                __VpreTriggered.andNot(vlSelf->__VactTriggered, vlSelf->__VnbaTriggered);
                vlSelf->__VnbaTriggered.set(vlSelf->__VactTriggered);
                VTile___024root___eval_act(vlSelf);
            }
        }
        if (vlSelf->__VnbaTriggered.any()) {
            __VnbaContinue = 1U;
            if (VL_UNLIKELY((0x64U < __VnbaIterCount))) {
#ifdef VL_DEBUG
                VTile___024root___dump_triggers__nba(vlSelf);
#endif
                VL_FATAL_MT("/home/digi/code/c/riscv-qv/Tile.sv", 3491, "", "NBA region did not converge.");
            }
            __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
            VTile___024root___eval_nba(vlSelf);
        }
    }
}

#ifdef VL_DEBUG
void VTile___024root___eval_debug_assertions(VTile___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VTile__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VTile___024root___eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((vlSelf->clock & 0xfeU))) {
        Verilated::overWidthError("clock");}
    if (VL_UNLIKELY((vlSelf->reset & 0xfeU))) {
        Verilated::overWidthError("reset");}
    if (VL_UNLIKELY((vlSelf->io_host_fromhost_valid 
                     & 0xfeU))) {
        Verilated::overWidthError("io_host_fromhost_valid");}
    if (VL_UNLIKELY((vlSelf->io_nasti_aw_ready & 0xfeU))) {
        Verilated::overWidthError("io_nasti_aw_ready");}
    if (VL_UNLIKELY((vlSelf->io_nasti_w_ready & 0xfeU))) {
        Verilated::overWidthError("io_nasti_w_ready");}
    if (VL_UNLIKELY((vlSelf->io_nasti_b_valid & 0xfeU))) {
        Verilated::overWidthError("io_nasti_b_valid");}
    if (VL_UNLIKELY((vlSelf->io_nasti_b_bits_id & 0xe0U))) {
        Verilated::overWidthError("io_nasti_b_bits_id");}
    if (VL_UNLIKELY((vlSelf->io_nasti_b_bits_resp & 0xfcU))) {
        Verilated::overWidthError("io_nasti_b_bits_resp");}
    if (VL_UNLIKELY((vlSelf->io_nasti_ar_ready & 0xfeU))) {
        Verilated::overWidthError("io_nasti_ar_ready");}
    if (VL_UNLIKELY((vlSelf->io_nasti_r_valid & 0xfeU))) {
        Verilated::overWidthError("io_nasti_r_valid");}
    if (VL_UNLIKELY((vlSelf->io_nasti_r_bits_id & 0xe0U))) {
        Verilated::overWidthError("io_nasti_r_bits_id");}
    if (VL_UNLIKELY((vlSelf->io_nasti_r_bits_resp & 0xfcU))) {
        Verilated::overWidthError("io_nasti_r_bits_resp");}
    if (VL_UNLIKELY((vlSelf->io_nasti_r_bits_last & 0xfeU))) {
        Verilated::overWidthError("io_nasti_r_bits_last");}
}
#endif  // VL_DEBUG
