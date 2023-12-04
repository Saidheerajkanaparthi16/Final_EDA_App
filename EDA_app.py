#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Disable deprecation warning from pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)

# Set the Seaborn style
sns.set_style("whitegrid")

def main():
    st.title("CSV EDA Web App")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head())

        # Basic Statistics
        if st.button('Show Statistics'):
            st.write(data.describe())

        # Correlation Heatmap
        if st.button('Show Correlation Heatmap'):
            numeric_data = data.select_dtypes(include=[np.number])
            non_binary_columns = numeric_data.nunique()[numeric_data.nunique() > 2].index.tolist()
            numeric_data = numeric_data[non_binary_columns]

            if numeric_data.empty:
                st.write("No suitable numeric columns available for correlation.")
            else:
                fig, ax = plt.subplots(figsize=(10, 10))
                corr_matrix = numeric_data.corr()
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", ax=ax)
                st.pyplot(fig)

        # Column selection for plotting and tests
        col_list = data.columns.tolist()
        selected_cols = st.multiselect("Select Columns for Visualization and Tests", col_list)

        # Visualization Types
        plot_types = ['Pair Plot', 'Joint Plot', 'Bar Plot', 'Count Plot']
        selected_plot = st.selectbox("Select Plot Type", plot_types)

        # Generate Visualization
        if st.button('Generate Visualization'):
            if not selected_cols:
                st.error("Please select at least one column.")
            else:
                try:
                    fig, ax = plt.subplots()

                    if selected_plot == 'Pair Plot':
                        sns.pairplot(data[selected_cols].dropna())
                        st.pyplot()

                    elif selected_plot == 'Joint Plot' and len(selected_cols) == 2:
                        sns.jointplot(data=data, x=selected_cols[0], y=selected_cols[1], kind="scatter")
                        st.pyplot()

                    elif selected_plot == 'Bar Plot' and len(selected_cols) > 0:
                        data[selected_cols].plot(kind='bar', ax=ax)
                        st.pyplot(fig)

                    elif selected_plot == 'Count Plot' and len(selected_cols) > 0:
                        for col in selected_cols:
                            sns.countplot(x=col, data=data)
                            st.pyplot()

                except TypeError:
                    st.error("Error: Selected column(s) are not suitable for this plot type.")

        # Statistical Tests
        if st.button('Perform Statistical Tests') and len(selected_cols) == 2:
            try:
                test_result = stats.pearsonr(data[selected_cols[0]].dropna(), data[selected_cols[1]].dropna())
                st.write(f"Pearson correlation coefficient: {test_result[0]:.2f}")
                st.write(f"P-value: {test_result[1]:.2e}")
            except TypeError:
                st.error("Error: Selected column(s) are not suitable for Pearson correlation test.")

        # Additional Visualizations
        visualization_types = ['Line Plot', 'Box Plot', 'Violin Plot', 'Stacked Bar Chart', 'Bubble Chart', 'Pie Chart', 'Heatmap']
        selected_viz = st.selectbox("Choose a Visualization", visualization_types)

        if st.button('Generate Additional Visualization'):
            if not selected_cols:
                st.error("Please select at least one column.")
            else:
                try:
                    fig, ax = plt.subplots()

                    if selected_viz == 'Line Plot':
                        for col in selected_cols:
                            ax.plot(data[col], label=col)
                        plt.legend()

                    elif selected_viz == 'Box Plot':
                        sns.boxplot(data=data[selected_cols])

                    elif selected_viz == 'Violin Plot':
                        sns.violinplot(data=data[selected_cols])

                    elif selected_viz == 'Stacked Bar Chart':
                        # This requires specific data handling
                        st.write("Stacked Bar Chart needs specific data selection.")

                    elif selected_viz == 'Bubble Chart':
                        # Assuming three columns: x, y, and bubble size
                        if len(selected_cols) == 3:
                            x, y, size = selected_cols
                            ax.scatter(data[x], data[y], s=data[size])
                        else:
                            st.error("Select exactly 3 columns for Bubble Chart.")

                    elif selected_viz == 'Pie Chart':
                        if len(selected_cols) == 1:
                            data[selected_cols[0]].value_counts().plot(kind='pie', autopct='%1.1f%%')
                        else:
                            st.error("Select exactly 1 column for Pie Chart.")

                    elif selected_viz == 'Heatmap':
                        sns.heatmap(data[selected_cols].corr(), annot=True, fmt=".2f")

                    st.pyplot(fig)
                except TypeError:
                    st.error("Error: Selected column(s) are not suitable for this plot type.")

if __name__ == "__main__":
    main()
