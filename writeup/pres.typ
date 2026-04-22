#align(center)[
  #text(22pt, weight: "bold")[Presentation Math]
  #v(0.3em)
  #text(14pt)[Gordon Chen]
]
#v(1em)

= Policy Gradient
$nabla_theta J(theta) = EE_(tau ~ pi_theta)[sum_t^T nabla_theta log pi_theta (a_t|s_t) Phi_t]$


= Advantage estimate
$Phi_t = v_t - V_phi.alt (s_t)$

$v_t = sum_(t'=t)^(T-1) gamma^(t'-t) r_t' + gamma^(T-t) V_phi.alt (s_T)$

$v_t = V_phi.alt (s_t) + sum_(t'=t)^(T-1) gamma^(t'-t) delta_t'$

$delta_t = r_t + gamma V_phi.alt (s_(t+1)) - V_phi.alt (s_t)$


= V-Trace
$v_t^mu = EE_(tau~mu) [V_phi.alt (s_t) + sum_(t'=t)^(T-1) gamma^(t'-t) delta_t']$

$v_t = EE_(tau~pi) [V_phi.alt (s_t) + sum_(t'=t)^(T-1) gamma^(t'-t) delta_t']$

$v_t = EE_(tau~mu) [V_phi.alt (s_t) + sum_(t'=t)^(T-1) gamma^(t'-t)
(product_(i=t)^t' (pi(a_t'|s_t')) / (mu(a_t'|s_t'))) delta_t']$

$v_t = EE_(tau ~ mu) [V_phi.alt (s_t) + sum_(t'=t)^(T-1) gamma^(t'-t) (product_(i=t)^t' rho_i) delta_t']$

$rho_t = (pi(a_t'|s_t')) / (mu(a_t'|s_t'))$

$hat(rho)_t = min (rho_t, rho_("max"))$

$hat(c)_t = min (rho_t, c_("max"))$

$v_t = V_phi.alt (s_t) + sum_(t'=t)^(T-1) gamma^(t'-t) (product_(i=t)^t' hat(c)_i) hat(rho)_t' delta_t'$


= Value Func update

$min med||v_t - V_phi.alt (s_t) ||^2$
