import { ActivityHandler, MessageFactory, TurnContext } from "botbuilder";
import { queryBackend } from "../services/apiClient";

export class IrccBot extends ActivityHandler {
  constructor() {
    super();

    this.onMessage(async (context: TurnContext, next) => {
      const question = context.activity.text || "";
      try {
        const result = await queryBackend({ question });
        const reply = `${result.answer}\n\nFormulaires: ${result.forms.join(", ")}\nSources: ${result.sources.join(", ")}`;
        await context.sendActivity(MessageFactory.text(reply));
      } catch (err) {
        await context.sendActivity(MessageFactory.text("Erreur lors de l'appel au moteur RAG."));
      }

      await next();
    });
  }
}
