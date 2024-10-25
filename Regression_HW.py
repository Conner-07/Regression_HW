
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score


diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_model = RandomForestRegressor(n_estimators=75, random_state=42)
linear_model = LinearRegression()
svr_model = SVR()


rf_model.fit(X_train, y_train)
linear_model.fit(X_train, y_train)
svr_model.fit(X_train, y_train)


y_pred_rf = rf_model.predict(X_test)
y_pred_lr = linear_model.predict(X_test)
y_pred_svr = svr_model.predict(X_test)


evs_rf = explained_variance_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

evs_lr = explained_variance_score(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

evs_svr = explained_variance_score(y_test, y_pred_svr)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)


print(f"Random Forest - Explained Variance Score: {evs_rf}, Mean Squared Error: {mse_rf}, R2 Score: {r2_rf}")
print(f"Linear Regression - Explained Variance Score: {evs_lr}, Mean Squared Error: {mse_lr}, R2 Score: {r2_lr}")
print(f"SVR - Explained Variance Score: {evs_svr}, Mean Squared Error: {mse_svr}, R2 Score: {r2_svr}")


best_model = max([('Random Forest', r2_rf), ('Linear Regression', r2_lr), ('SVR', r2_svr)], key=lambda x: x[1])
print(f"The best performing model is: {best_model[0]} with an R2 score of {best_model[1]:.2f}")


