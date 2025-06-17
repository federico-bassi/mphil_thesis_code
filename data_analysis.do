/*********************************************************************
* Root folders (set once)
*********************************************************************/
local root     "/Users/federicobassi/Desktop/TI.nosync/MPhil_Thesis"
local datadir  "`root'/data"
local img_path "`root'/plots"


clear
import excel using "`datadir'/garp.xlsx", firstrow clear
preserve                      
keep if direction == -1
save "`datadir'/garp_ds.dta", replace   
restore                        
preserve                      
keep if direction == 1
save "`datadir'/garp_us.dta", replace   
restore                       


* ------------------------------------------------------------------
* DOWNWARD SLOPING 
* ------------------------------------------------------------------
use "`datadir'/garp_ds.dta", clear
gen double price = max_self/max_other
gen double g = (price * noisy_y) / (price * noisy_y + noisy_x)
encode pucktreatment, gen(treatment)
egen constraint = group(max_self max_other), label
reghdfe g i.treatment, absorb(constraint) vce(cluster id)

coefplot, drop(_cons constraint*)                               ///
        xline(0) title("A&M constraints – FE for budget")          ///
        yscale(reverse) ylabel(, angle(horizontal))


* ------------------------------------------------------------------
* UPWARD SLOPING 
* ------------------------------------------------------------------
use "`datadir'/garp_us.dta", clear

gen double left = (noisy_x-max_self)/max_self
encode pucktreatment, gen(treatment)
egen constraint = group(max_self max_other), label

reghdfe left i.treatment, absorb(constraint) vce(cluster id)

coefplot, drop(_cons constraint*)                               ///
        xline(0) title("A&M constraints – FE for budget")          ///
        yscale(reverse) ylabel(, angle(horizontal))
