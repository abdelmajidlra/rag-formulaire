import * as restify from "restify";
import { BotFrameworkAdapter } from "botbuilder";

import { IrccBot } from "./bot/IrccBot";

const adapter = new BotFrameworkAdapter({
  appId: process.env.MicrosoftAppId,
  appPassword: process.env.MicrosoftAppPassword,
});

adapter.onTurnError = async (context, error) => {
  console.error(`[onTurnError] unhandled error: ${error}`);
  await context.sendActivity("Le bot a rencontré une erreur.");
};

const server = restify.createServer();
server.use(restify.plugins.bodyParser());
server.listen(process.env.PORT || 3978, () => {
  console.log(`\nBot démarré sur le port ${process.env.PORT || 3978}`);
});

const bot = new IrccBot();

server.post("/api/messages", (req, res) => {
  adapter.processActivity(req, res, async (context) => {
    await bot.run(context);
  });
});
