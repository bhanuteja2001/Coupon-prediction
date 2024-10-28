This configuration file is used by Prefect to identify and manage the workflow's main flows. It specifies:
The primary workflow flow, named "main-flow", which is defined in the 'model.py' file and uses the 'main_flow' function.
A specialized flow for coupon acceptance prediction, named "main-flow-coupon-accepting", also defined in 'model.py' but using the 'main_flow_coupon_accepting' function.
These definitions allow Prefect to locate and execute the appropriate Python functions when running the workflow. 