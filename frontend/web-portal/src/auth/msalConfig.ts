export const msalConfig = {
  auth: {
    clientId: "YOUR_CLIENT_ID",
    authority: "https://login.microsoftonline.com/YOUR_TENANT_ID",
    redirectUri: "/",
  },
  cache: {
    cacheLocation: "localStorage" as const,
  },
};

export const loginRequest = {
  scopes: ["api://your-api-id/.default"],
};
