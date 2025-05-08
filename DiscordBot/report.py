# report.py
from enum import Enum, auto
import discord
from discord.ui import View, Button, Select
import re
import datetime

class State(Enum):
    REPORT_START = auto()
    AWAITING_MESSAGE = auto()
    MESSAGE_IDENTIFIED = auto()
    REPORT_TYPE = auto()
    HATE_SPEECH_TYPE = auto()
    HATE_SPEECH_TARGETED = auto()
    HATE_SPEECH_CONTEXT = auto()
    BLOCK_USER = auto()
    REPORT_COMPLETE = auto()

class Report:
    START_KEYWORD = "report"
    CANCEL_KEYWORD = "cancel"
    HELP_KEYWORD = "help"

    # Report types
    VIOLENT_THREAT = "violent threat"
    SPAM = "spam"
    OBSCENE_CONTENT = "unsolicited obscene content (not csam)"
    CSAM = "csam"
    HATE_SPEECH = "hate speech"

    # Hate speech types
    HS_RACISM = "racism"
    HS_SEXISM = "sexism"
    HS_HOMOPHOBIA = "homophobia"
    HS_ABLEISM = "ableism"
    HS_RELIGIOUS = "religious discrimination"
    HS_OTHER = "other"

    def __init__(self, client):
        self.state = State.REPORT_START
        self.client = client
        self.message = None
        self.report_type = None
        self.hate_speech_type = None
        self.hate_speech_targeted = None
        self.context_messages = 0
        self.reported_user = None
        self.last_interaction = None  # To store the last message with components

    async def handle_message(self, message):
        '''
        This function makes up the meat of the user-side reporting flow. It defines how we transition between states and what
        prompts to offer at each of those states.
        '''

        if message.content.lower() == self.CANCEL_KEYWORD:
            self.state = State.REPORT_COMPLETE
            return ["Report cancelled."]

        if message.content.lower() == self.HELP_KEYWORD:
            reply = "**Reporting Process Help**\n"
            reply += "- Use `report` to start reporting a message\n"
            reply += "- Use `cancel` to cancel the current report\n"
            reply += "- Follow the prompts and use the buttons/menus to complete your report\n"
            reply += "- Your report will be reviewed by our moderation team"
            return [reply]

        if self.state == State.REPORT_START:
            reply = "**Report Abuse**\n\n"
            reply += "Thank you for starting the reporting process. "
            reply += "Say `help` at any time for more information.\n\n"
            reply += "Please copy paste the link to the message you want to report.\n"
            reply += "You can obtain this link by right-clicking the message and clicking `Copy Message Link`."
            self.state = State.AWAITING_MESSAGE
            return [reply]

        if self.state == State.AWAITING_MESSAGE:
            # Parse out the three ID strings from the message link
            m = re.search('/(\d+)/(\d+)/(\d+)', message.content)
            if not m:
                return ["I'm sorry, I couldn't read that link. Please try again or say `cancel` to cancel."]
            guild = self.client.get_guild(int(m.group(1)))
            if not guild:
                return ["I cannot accept reports of messages from guilds that I'm not in. Please have the guild owner add me to the guild and try again."]
            channel = guild.get_channel(int(m.group(2)))
            if not channel:
                return ["It seems this channel was deleted or never existed. Please try again or say `cancel` to cancel."]
            try:
                self.message = await channel.fetch_message(int(m.group(3)))
                self.reported_user = self.message.author
            except discord.errors.NotFound:
                return ["It seems this message was deleted or never existed. Please try again or say `cancel` to cancel."]

            self.state = State.MESSAGE_IDENTIFIED

            reply = "**Report Abuse**\n\n"
            reply += f"I found this message from {self.message.author.name}:\n"

            # Handle both text messages and other content types (like audio)
            if self.message.content:
                reply += f"```{self.message.content}```\n"
            elif len(self.message.attachments) > 0:
                reply += f"*[Message contains attachments/media]*\n"
            else:
                reply += f"*[Message has no text content]*\n"

            reply += "What type of violation are you reporting?"

            # Create a view with buttons for report type selection
            view = ReportTypeView(self)

            self.state = State.REPORT_TYPE
            return [(reply, view)]

        # Handle "Other" hate speech type specification
        if self.state == State.HATE_SPEECH_TYPE and self.hate_speech_type == self.HS_OTHER and message.content:
            # Store the user's specification
            self.hate_speech_type = f"{self.HS_OTHER}: {message.content}"
            self.state = State.HATE_SPEECH_TARGETED

            # Create yes/no buttons for targeted question
            view = TargetedView(self)

            return [(
                "**Other Hate Speech**\n\nWas this targeted towards an individual in the Discord server?",
                view
            )]

        # Handle text input for context messages (only part that still requires typing)
        if self.state == State.HATE_SPEECH_CONTEXT and message.content:
            try:
                # Try to parse as integer, default to 0 if not a number
                self.context_messages = int(message.content.strip()) if message.content.strip().isdigit() else 0
            except ValueError:
                self.context_messages = 0

            self.state = State.BLOCK_USER

            reply = "**Block User?**\n\n"
            reply += "Submitted. Thank you for your report. Our content moderation team will review the messages and decide on the appropriate action, which may include post and/or account removal. Would you like to block the user?"

            # Create block user view
            view = BlockUserView(self)

            return [(reply, view)]

        return ["I'm sorry, I couldn't process your request. Please try again or say `cancel` to cancel the report."]

    def report_complete(self):
        return self.state == State.REPORT_COMPLETE

    def get_report_data(self):
        """
        Returns a dictionary with all the report data for storing or processing.
        """
        return {
            "reported_message": self.message.id if self.message else None,
            "reported_user": self.reported_user.id if self.reported_user else None,
            "report_type": self.report_type,
            "hate_speech_type": self.hate_speech_type,
            "hate_speech_targeted": self.hate_speech_targeted,
            "context_messages": self.context_messages,
            "timestamp": datetime.datetime.now().isoformat()
        }

class ReportTypeView(View):
    def __init__(self, report):
        super().__init__(timeout=300)  # 5 minute timeout
        self.report = report

        # Add buttons for each report type
        self.add_item(Button(style=discord.ButtonStyle.danger, label="Violent Threat", custom_id="violent_threat"))
        self.add_item(Button(style=discord.ButtonStyle.secondary, label="Spam", custom_id="spam"))
        self.add_item(Button(style=discord.ButtonStyle.secondary, label="Obscene Content", custom_id="obscene"))
        self.add_item(Button(style=discord.ButtonStyle.danger, label="CSAM", custom_id="csam"))
        self.add_item(Button(style=discord.ButtonStyle.secondary, label="Hate Speech", custom_id="hate_speech"))

    async def interaction_check(self, interaction):
        # Map button custom_id to report types
        report_map = {
            "violent_threat": Report.VIOLENT_THREAT,
            "spam": Report.SPAM,
            "obscene": Report.OBSCENE_CONTENT,
            "csam": Report.CSAM,
            "hate_speech": Report.HATE_SPEECH
        }

        self.report.report_type = report_map[interaction.data["custom_id"]]

        try:
            # Handle different report types
            if self.report.report_type == Report.VIOLENT_THREAT:
                self.report.state = State.REPORT_COMPLETE
                await interaction.response.send_message("**Submitted**\n\nThank you for your report. The user has been blocked and their activity will now be more closely monitored. If you believe the violent threat to be legitimate please contact local law enforcement.")

            elif self.report.report_type in [Report.SPAM, Report.OBSCENE_CONTENT]:
                self.report.state = State.REPORT_COMPLETE
                await interaction.response.send_message("**Submitted**\n\nThank you for your report. The user has been blocked and their activity will now be more closely monitored.")

            elif self.report.report_type == Report.CSAM:
                self.report.state = State.REPORT_COMPLETE
                await interaction.response.send_message("**Submitted**\n\nThank you for your report. The user has been blocked and their activity will be reported to the police. No further action is required.")

            elif self.report.report_type == Report.HATE_SPEECH:
                self.report.state = State.HATE_SPEECH_TYPE

                # Create hate speech type selection menu
                view = HateSpeechTypeView(self.report)
                await interaction.response.send_message(
                    "**Hate Speech**\n\nPlease specify the type of hate speech:",
                    view=view
                )
        except discord.errors.InteractionResponded:
            # If the interaction was already responded to, use followup
            if self.report.report_type == Report.HATE_SPEECH:
                view = HateSpeechTypeView(self.report)
                await interaction.followup.send(
                    "**Hate Speech**\n\nPlease specify the type of hate speech:",
                    view=view
                )

        return True

class HateSpeechTypeView(View):
    def __init__(self, report):
        super().__init__(timeout=300)
        self.report = report

        # Create dropdown for hate speech types
        options = [
            discord.SelectOption(label="Racism", value=Report.HS_RACISM),
            discord.SelectOption(label="Sexism", value=Report.HS_SEXISM),
            discord.SelectOption(label="Homophobia", value=Report.HS_HOMOPHOBIA),
            discord.SelectOption(label="Ableism", value=Report.HS_ABLEISM),
            discord.SelectOption(label="Religious discrimination", value=Report.HS_RELIGIOUS),
            discord.SelectOption(label="Other", value=Report.HS_OTHER, description="Please specify after selecting")
        ]

        # Add the dropdown to the view
        select_menu = Select(
            placeholder="Select hate speech type...",
            options=options
        )
        select_menu.callback = self.on_select
        self.add_item(select_menu)

    async def on_select(self, interaction):
        selected_type = interaction.data["values"][0]
        self.report.hate_speech_type = selected_type

        # If "Other" is selected, ask for specification
        if selected_type == Report.HS_OTHER:
            await interaction.response.send_message(
                "**Other Hate Speech**\n\nPlease specify the type of hate speech. We will take it just as seriously."
            )
            # Stay in the same state to wait for the text input
            return

        # Move to the next state
        self.report.state = State.HATE_SPEECH_TARGETED

        # Format the header based on the hate speech type
        header = self.report.hate_speech_type.title()

        # Create yes/no buttons for targeted question
        view = TargetedView(self.report)
        await interaction.response.send_message(
            f"**{header}**\n\nWas this targeted towards an individual in the Discord server?",
            view=view
        )

class TargetedView(View):
    def __init__(self, report):
        super().__init__(timeout=300)
        self.report = report

        # Add yes/no buttons
        self.add_item(Button(style=discord.ButtonStyle.success, label="Yes", custom_id="yes"))
        self.add_item(Button(style=discord.ButtonStyle.danger, label="No", custom_id="no"))

    async def interaction_check(self, interaction):
        self.report.hate_speech_targeted = interaction.data["custom_id"] == "yes"
        self.report.state = State.HATE_SPEECH_CONTEXT

        # Format the header based on the hate speech type
        if ":" in self.report.hate_speech_type:
            header = "Hate Speech"
        else:
            header = self.report.hate_speech_type.title()

        try:
            await interaction.response.send_message(
                f"**{header}**\n\n(Optional) If any, how many messages before this one should we include as context in assessing this report? Voice memos and audio files are included.\n\n" +
                "Please reply with a number."
            )
        except discord.errors.InteractionResponded:
            # If the interaction was already responded to, use followup
            await interaction.followup.send(
                f"**{header}**\n\n(Optional) If any, how many messages before this one should we include as context in assessing this report? Voice memos and audio files are included.\n\n" +
                "Please reply with a number."
            )

        return True

class BlockUserView(View):
    def __init__(self, report):
        super().__init__(timeout=300)
        self.report = report

        # Add yes/no buttons
        self.add_item(Button(style=discord.ButtonStyle.success, label="Yes, block user", custom_id="yes"))
        self.add_item(Button(style=discord.ButtonStyle.secondary, label="No, don't block", custom_id="no"))

    async def interaction_check(self, interaction):
        should_block = interaction.data["custom_id"] == "yes"

        # SIMULATE BLOCKING THE USER
        block_result = "You have chosen to block the user." if should_block else "You have chosen not to block the user."

        self.report.state = State.REPORT_COMPLETE

        try:
            await interaction.response.send_message(
                f"**Report Complete**\n\n{block_result} Your report has been submitted and will be reviewed by our moderation team."
            )
        except discord.errors.InteractionResponded:
            # If the interaction was already responded to, use followup
            await interaction.followup.send(
                f"**Report Complete**\n\n{block_result} Your report has been submitted and will be reviewed by our moderation team."
            )

        return True


class ModerationReviewView(View):
    def __init__(self, report, bot):
        super().__init__(timeout=600)
        self.report = report
        self.bot = bot

        self.add_item(Button(style=discord.ButtonStyle.secondary, label="Ignore", custom_id="ignore"))
        self.add_item(Button(style=discord.ButtonStyle.primary, label="Warn", custom_id="warn"))
        self.add_item(Button(style=discord.ButtonStyle.danger, label="Punish", custom_id="punish"))
        self.add_item(Button(style=discord.ButtonStyle.danger, label="Escalate", custom_id="escalate"))

    async def interaction_check(self, interaction):
        action = interaction.data["custom_id"]
        
        if action == "ignore":
            await interaction.response.send_message("**Action Taken: Ignored**\n\n(Optional) Leave a message explaining your decision.", ephemeral=True)

        elif action == "warn":
            await interaction.response.send_message("**Action Taken: Warned the user.**\n\n(Optional) Leave a message explaining your decision.", ephemeral=True)

        elif action == "punish":
            view = PunishActionView(self.report, self.bot)
            await interaction.response.send_message("🔨 **Punishment Options:**\nSelect an action below:", view=view, ephemeral=True)

        elif action == "escalate":
            view = EscalationView(self.report, self.bot)
            await interaction.response.send_message("🚩 **Escalation Options:**\nSelect escalation target:", view=view, ephemeral=True)

        return True

class PunishActionView(View):
    def __init__(self, report, bot):
        super().__init__(timeout=600)
        self.report = report
        self.bot = bot

        self.add_item(Button(style=discord.ButtonStyle.danger, label="Remove Message", custom_id="remove"))
        self.add_item(Button(style=discord.ButtonStyle.danger, label="Temporary Mute", custom_id="mute"))
        self.add_item(Button(style=discord.ButtonStyle.danger, label="Shadow Ban", custom_id="shadowban"))
        self.add_item(Button(style=discord.ButtonStyle.danger, label="Kick User", custom_id="kick"))

    async def interaction_check(self, interaction):
        punishment = interaction.data["custom_id"]

        actions = {
            "remove": "Message removed.",
            "mute": "User temporarily muted.",
            "shadowban": "User shadow-banned.",
            "kick": "User kicked from server."
        }

        result = actions.get(punishment, "Action performed.")

        await interaction.response.send_message(f"**Action Taken: {result}**\n(Optional) Leave a message explaining your decision.", ephemeral=True)

        return True

class EscalationView(View):
    def __init__(self, report, bot):
        super().__init__(timeout=600)
        self.report = report
        self.bot = bot

        self.add_item(Button(style=discord.ButtonStyle.primary, label="Trust and Safety Team", custom_id="trust"))
        self.add_item(Button(style=discord.ButtonStyle.danger, label="Law Enforcement", custom_id="law"))

    async def interaction_check(self, interaction):
        target = interaction.data["custom_id"]

        targets = {
            "trust": "Escalated to Trust and Safety team.",
            "law": "Escalated to Law Enforcement."
        }

        result = targets.get(target, "Escalated.")

        await interaction.response.send_message(f"**{result}**\n(Required) Why did you escalate the report?", ephemeral=True)

        return True

