import streamlit as st
import numpy as np
from scipy.optimize import minimize
from PIL import Image
import base64
from io import BytesIO

# -------------------------
# Calculation Functions
# -------------------------
def calculate_imposter_joint(force_value):
    try:
        P = force_value
        MaxLoad = P * 4.4482216
        DIA = 9.53

        MUS = 450  # Member ultimate strength
        MYS = 350  # Member yield strength
        FA = 3.18  # Fabrication accuracy

        # Given constants
        shearlag = 0.60
        MMCU = 0.10  # Maximum moment at connection (unreleased)
        nboltsinline = 1
        boltspacing = 0
        shearplanes = 2
        Ks = 0.3
        Cs = 1
        Fub = 825
        phibr = 0.8
        phib = 0.8
        phiu = 0.75
        phi = 0.9

        def calculate_DSF(params):
            T1, T2, T3, W, E, L, G = params
            slopetaper = ((T2 / 2) - (T1 / 2)) / L
            boltradius = DIA / 2
            boltarea = np.pi * (boltradius ** 2)
            boltlengthend = L - G - E - (DIA + 2)
            minthicknessmale = T1 + (2 * slopetaper * boltlengthend)
            NetArea = min((T1 * (W - nboltsinline * (2 + DIA))), (W * ((T3 - T2) + 2 * G * slopetaper)))
            if NetArea <= 0:
                return -np.inf, -np.inf, -np.inf, -np.inf
            avg_T1_term = np.average([T1, T1 + 2 * slopetaper * boltlengthend])
            avg_T3_term = np.average([(T3 - T2) + 2 * G * slopetaper, (T3 - T2) + 2 * (G + boltlengthend) * slopetaper])
            BSnetsheararea = min(boltlengthend * avg_T1_term, E * avg_T3_term)
            if BSnetsheararea <= 0:
                return -np.inf, -np.inf, -np.inf, -np.inf
            BSnettensionarea = boltspacing * (nboltsinline - 1)
            Z = min((minthicknessmale / 2) * minthicknessmale * W, minthicknessmale * W * (W / 2))
            Mp = Z * (MYS / 1000)
            Vr = 0.7 * 0.6 * phib * nboltsinline * shearplanes * boltarea * Fub / 1000
            Br = 3 * phibr * nboltsinline * minthicknessmale * DIA * MUS / 1000
            Tr = phiu * shearlag * NetArea * MUS / 1000
            Vb = phiu * (1 * BSnettensionarea * MUS + (0.6 * BSnetsheararea * ((MYS + MUS) / 2) * 1 / 1000))
            DSF1 = Vr / MaxLoad
            DSF2 = Br / MaxLoad
            DSF3 = Tr / MaxLoad
            DSF4 = Vb / MaxLoad
            return DSF1, DSF2, DSF3, DSF4

        def objective(params):
            DSF1, DSF2, DSF3, DSF4 = calculate_DSF(params)
            if DSF1 == -np.inf or DSF2 == -np.inf or DSF3 == -np.inf or DSF4 == -np.inf:
                return np.inf  # Penalize invalid configurations
            return (DSF1 - 1.3) ** 2 + (DSF2 - 1.25) ** 2 + (DSF3 - 1.25) ** 2 + (DSF4 - 1.25) ** 2

        def constraint_DSF1(params):
            DSF1, _, _, _ = calculate_DSF(params)
            return DSF1 - 1.251

        def constraint_DSF2(params):
            _, DSF2, _, _ = calculate_DSF(params)
            return DSF2 - 1.25

        def constraint_DSF3(params):
            _, _, DSF3, _ = calculate_DSF(params)
            return DSF3 - 1.25

        def constraint_DSF4(params):
            _, _, _, DSF4 = calculate_DSF(params)
            return DSF4 - 1.25

        def constraint_T1_T2(params):
            T1, T2, _, _, _, _, _ = params
            epsilon1 = 2 * T1  # Enforcing Factor to bind Imposter Joint
            return T2 - T1 - epsilon1

        def constraint_T2_T3(params):
            _, T2, T3, _, _, _, _ = params
            epsilon2 = 2 * T2  # Enforcing Factor to bind Imposter Joint
            return T3 - T2 - epsilon2

        bounds = [
            (0.794, 50.0),  # T1 bounds
            (0.794, 50.0),  # T2 bounds
            (0.794, 50.0),  # T3 bounds
            (0.794, 70.0),  # W bounds
            (0.794, 25.0),  # E bounds
            (0.794, 70.0),  # L bounds
            (0.794, 5.0)    # G bounds
        ]
        initial_guess = [8.0, 18.0, 22.0, 95.0, 12.0, 28.0, 0.5]
        constraints = [
            {'type': 'ineq', 'fun': constraint_DSF1},
            {'type': 'ineq', 'fun': constraint_DSF2},
            {'type': 'ineq', 'fun': constraint_DSF3},
            {'type': 'ineq', 'fun': constraint_DSF4},
            {'type': 'ineq', 'fun': constraint_T1_T2},
            {'type': 'ineq', 'fun': constraint_T2_T3}
        ]
        result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints, method='SLSQP')
        optimized_params = result.x
        DSF1, DSF2, DSF3, DSF4 = calculate_DSF(optimized_params)
        output = (
            f"Optimized Imposter Joint Parameters:\n\n"
            f"Minimum Thickness of Male (T1): {optimized_params[0]:.3f}\n"
            f"Maximum Thickness of Male (T2): {optimized_params[1]:.3f}\n"
            f"Overall Thickness (T3): {optimized_params[2]:.3f}\n"
            f"Minimum Width at Connection (W): {optimized_params[3]:.3f}\n"
            f"Distance to the End (E): {optimized_params[4]:.3f}\n"
            f"Length of Male (L): {optimized_params[5]:.3f}\n"
            f"Gap (G): {optimized_params[6]:.3f}\n"
            f"Bolt Diameter (DIA): {DIA:.3f}\n\n"
            f"DSF1: {DSF1:.3f}, DSF2: {DSF2:.3f}, DSF3: {DSF3:.3f}, DSF4: {DSF4:.3f}"
        )
        return output
    except Exception as e:
        return f"Error in calculation: {e}"


def calculate_u_joint(force_value):
    try:
        
        # Constants
        P = force_value  # max axial load
        p = P * 4.4482216
        MUS = 450  # Maximum Ultimate Strength
        MYS = 350  # Maximum Yield Strength
        FT = 1  # Fraction of Thickness at EOM
        ShearLag = 0.6
        BoltDiameter = 9.53  # Fixed Bolt Diameter

        # Joint Design Inputs
        n = 1  # Number of Bolts in Line
        boltspacing = 0
        ShearPlanes = 2
        Ks = 0.30  # Mean Slip Coefficient based on Surface
        Cs = 1  # Coefficient of 5% slip probability

        # Known Parameters
        Fub = 825
        Phibr = 0.8
        Phib = 0.8
        Phiu = 0.75

        # Function to calculate DSFs
        def calculate_DSF(params):
            t, w, l = params
            
            
            boltRadius = BoltDiameter / 2
            boltArea = np.pi * (boltRadius ** 2)
            NetAreaConn = t*(w-n*(4+2*boltRadius)) #Net Area Connection
            
            # Area Check
            if NetAreaConn <= 0:
                return -np.inf, -np.inf, -np.inf, -np.inf  
            
            BSnetShearArea = t*(l+2*boltRadius+4)
            BSnetTensionArea = boltspacing*(n-1)
            
            # Capacities
            Vr = 0.7 * 0.6 * Phib * n * ShearPlanes * boltArea * Fub / 1000  # Bolt Shear Capacity
            Br = 3 * Phibr * n * t * BoltDiameter * MUS / 1000  # Bearing Capacity
            Tr = Phiu * ShearLag * NetAreaConn * MUS / 1000  # Net Section Rupture
            BlockShear = Phiu * (1*BSnetTensionArea * MUS + (0.6 * BSnetShearArea * (MUS + MYS) / 2)) / 1000  # Block Shear

            # Design Safety Factors
            DSF1 = Vr / p
            DSF2 = Br / p
            DSF3 = Tr / p
            DSF4 = BlockShear / p
            
            return DSF1, DSF2, DSF3, DSF4

        # Objective function to minimize the squared deviations from 1.3
        def objective(params):
            DSF1, DSF2, DSF3, DSF4 = calculate_DSF(params)
            if DSF1 == -np.inf or DSF2 == -np.inf or DSF3 == -np.inf or DSF4 == -np.inf:
                return np.inf  # Penalize invalid configurations
            
            # Deviation metrics from 1.26 for each DSF
            dsf_penalty = (DSF1 - 1.26)**2 + (DSF2 - 1.26)**2 + (DSF3 - 1.26)**2 + (DSF4 - 1.26)**2
            return dsf_penalty

        # Constraints to ensure each DSF >= 1.3
        constraints = [
            {'type': 'ineq', 'fun': lambda params: calculate_DSF(params)[0] - 1.26},  # DSF1
            {'type': 'ineq', 'fun': lambda params: calculate_DSF(params)[1] - 1.26},  # DSF2
            {'type': 'ineq', 'fun': lambda params: calculate_DSF(params)[2] - 1.26},  # DSF3
            {'type': 'ineq', 'fun': lambda params: calculate_DSF(params)[3] - 1.26}   # DSF4
        ]

        # Define bounds for t, w, l
        bounds = [
            (0.794, 100),  # t: thickness at connection, hardset from 2 to avoid boundary limit
            (0.794, 100),  # w: width at connection
            (0.794, 100)   # l: length to end
        ]

        
        initial_guess = [9.53, 26.98, 19.05]  # feasible region

        # Run optimization
        result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints, method='trust-constr')

    
        optimized_params = result.x
        DSF1, DSF2, DSF3, DSF4 = calculate_DSF(optimized_params)
        output = (
            f"Optimized Parameters (t, w, l):\n\n"
            f"Minimum Thickness at Connection (t): {optimized_params[0]:.3f}\n"
            f"Minimum Width at Connection (w): {optimized_params[1]:.3f}\n"
            f"Minimum Distance to End (l): {optimized_params[2]:.3f}\n"
            f"Bolt Diameter (DIA): {BoltDiameter:.3f}\n\n"
            f"DSF1: {DSF1:.3f}, DSF2: {DSF2:.3f}, DSF3: {DSF3:.3f}, DSF4: {DSF4:.3f}"
        )
        return output
    except Exception as e:
        return f"Error in calculation: {e}"


def calculate_plate_joint(force_value):
    try:
        
        # Constants
        P = force_value  # max axial load
        p = P * 4.4482216
        MUS = 450  # Maximum Ultimate Strength
        MYS = 350  # Maximum Yield Strength
        FT = 1  # Fraction of Thickness at EOM
        ShearLag = 0.6
        BoltDiameter = 9.53  # Fixed Bolt Diameter

        # Joint Design Inputs
        n = 1  # Number of Bolts in Line
        boltspacing = 0
        ShearPlanes = 1
        Ks = 0.30  # Mean Slip Coefficient based on Surface
        Cs = 1  # Coefficient of 5% slip probability

        # Known Parameters
        Fub = 825
        Phibr = 0.8
        Phib = 0.8
        Phiu = 0.75

        # calculate DSFs
        def calculate_DSF(params):
            t, w, l = params  # Unpack
            
            # Intermediate Calculations
            boltRadius = BoltDiameter / 2
            boltArea = np.pi * (boltRadius ** 2)
            NetAreaConn = t * (w - n * (2 + BoltDiameter))  # Net Area Connection
            
            # Check for valid areas
            if NetAreaConn <= 0:
                return -np.inf, -np.inf, -np.inf, -np.inf  # Return invalid DSF if area is zero or negative
            
            meanv1 = FT * t
            meanV2 = t
            average = (meanv1 + meanV2) / 2
            BSnetShearArea = average * l * 2
            BSnetTensionArea = boltspacing*(n-1)
            
            # Capacities
            Vr = 0.7 * 0.6 * Phib * n * ShearPlanes * boltArea * Fub / 1000  # Bolt Shear Capacity
            Br = 3 * Phibr * n * t * BoltDiameter * MUS / 1000  # Bearing Capacity
            Tr = Phiu * ShearLag * NetAreaConn * MUS / 1000  # Net Section Rupture
            BlockShear = Phiu * (1*BSnetTensionArea * MUS + (0.6 * BSnetShearArea * (MUS + MYS) / 2)) / 1000  # Block Shear

            # Design Safety Factors
            DSF1 = Vr / p
            DSF2 = Br / p
            DSF3 = Tr / p
            DSF4 = BlockShear / p
            
            return DSF1, DSF2, DSF3, DSF4

        # Objective function to minimize the squared deviations from 1.3
        def objective(params):
            DSF1, DSF2, DSF3, DSF4 = calculate_DSF(params)
            if DSF1 == -np.inf or DSF2 == -np.inf or DSF3 == -np.inf or DSF4 == -np.inf:
                return np.inf  # Penalize invalid configurations
            
            # Base objective: deviation metrics from 1.26 for each DSF
            dsf_penalty = (DSF1 - 1.26)**2 + (DSF2 - 1.26)**2 + (DSF3 - 1.26)**2 + (DSF4 - 1.26)**2
            return dsf_penalty

        
        constraints = [
            {'type': 'ineq', 'fun': lambda params: calculate_DSF(params)[0] - 1.26},  # DSF1
            {'type': 'ineq', 'fun': lambda params: calculate_DSF(params)[1] - 1.26},  # DSF2
            {'type': 'ineq', 'fun': lambda params: calculate_DSF(params)[2] - 1.26},  # DSF3
            {'type': 'ineq', 'fun': lambda params: calculate_DSF(params)[3] - 1.26}   # DSF4
        ]

        # Define bounds for t, w, l
        bounds = [
            (0.794, 100),  # t: thickness at connection, hardset from 2 to avoid boundary limit
            (0.794, 100),  # w: width at connection
            (0.794, 100)   # l: length to end
        ]

        # Initial guess for t, w, l
        initial_guess = [6, 18, 8]  # feasible region

        # Run optimization
        result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints, method='trust-constr')

        
        optimized_params = result.x
        DSF1, DSF2, DSF3, DSF4 = calculate_DSF(optimized_params)
        output = (
            f"Optimized Parameters in millimeters (t, w, l):\n\n"
            f"Minimum thickness at Connection (t): {optimized_params[0]:.3f}\n"
            f"Minimum width at Connection (w): {optimized_params[1]:.3f}\n"
            f"Minimum Distance to the end (l): {optimized_params[2]:.3f}\n"
            f"Bolt Diameter (DIA): {BoltDiameter:.3f}\n\n"
            f"DSF1: {DSF1:.3f}, DSF2: {DSF2:.3f}, DSF3: {DSF3:.3f}, DSF4: {DSF4:.3f}\n\n"
            "Warning: Axial force exceeding 3.8 kips are beyond constraint threshold. "
            "To maintain a safety factor of 1.25, please consider selecting an alternative joint."
        )
        return output
    except Exception as e:
        return f"Error in calculation: {e}"


# -------------------------
# Streamlit App Layout
# -------------------------
def main():
    st.title("Joint Sizing Optimizer")
    # Sidebar for navigation between pages
    menu = st.sidebar.radio("Navigation", ("Home", "Imposter Joint", "U Joint", "Plate Joint"))

    if menu == "Home":
        try:
            logo = Image.open("SB-Full-Colour-Grey.png")
            st.image(logo, width=700)
        except Exception:
            st.write("Logo image not available.")
        st.header("Joint Sizing Optimization")
        st.write("Select a joint type from the sidebar to proceed.")

    elif menu == "Imposter Joint":
        st.header("Imposter Joint - Max Axial Force Input")
        force_value = st.number_input("Enter max axial force for Imposter Joint (in kips):", value=1.0, step=0.1)
        if st.button("Calculate", key="calc_imposter"):
            result = calculate_imposter_joint(force_value)
            st.text(result)
            try:
                imp_image = Image.open("Imposter.png")
                st.image(imp_image, width=700)
            except Exception:
                st.write("Imposter joint image not available.")
        if st.button("Reset", key="back_imposter"):
            st.rerun()

    elif menu == "U Joint":
        st.header("U Joint - Max Axial Force Input")
        force_value = st.number_input("Enter max axial force for U Joint (in kips):", value=1.0, step=0.1)
        if st.button("Calculate", key="calc_u"):
            result = calculate_u_joint(force_value)
            st.text(result)
            try:
                u_image = Image.open("UJoint.png")
                st.image(u_image, width=400)
            except Exception:
                st.write("U Joint image not available.")
        if st.button("Reset", key="back_u"):
            st.rerun()

    elif menu == "Plate Joint":
        st.header("Plate Joint - Max Axial Force Input")
        force_value = st.number_input("Enter max axial force for Plate Joint (in kips):", value=1.0, step=0.1)
        if st.button("Calculate", key="calc_plate"):
            result = calculate_plate_joint(force_value)
            st.text(result)
            try:
                plate_image = Image.open("PlateJoint.png")
                st.image(plate_image, width=700)
            except Exception:
                st.write("Plate Joint image not available.")
        if st.button("Reset", key="back_plate"):
            st.rerun()


if __name__ == "__main__":
    main()
