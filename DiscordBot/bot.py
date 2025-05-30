# bot.py
import discord
from discord.ext import commands
import os
import json
import logging
import re
import requests
from report import Report, ReportTypeView, HateSpeechTypeView, TargetedView, BlockUserView, State, ModerationReviewView, PunishActionView, EscalationView
import datetime
import asyncio
from typing import List

from model import RacistClassifier, Message

# Set up logging to the console
logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)

# Stream handler to see logs in console output
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(console_handler)

# There should be a file called 'tokens.json' inside the same folder as this file
token_path = 'tokens.json'
if not os.path.isfile(token_path):
    raise Exception(f"{token_path} not found!")
with open(token_path) as f:
    # If you get an error here, it means your token is formatted incorrectly. Did you put it in quotes?
    tokens = json.load(f)
    discord_token = tokens['discord']

class ModBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True
        super().__init__(intents=intents)
        self.group_num = None
        self.mod_channels = {}
        self.reports = {}
        self.completed_reports = []
        self.active_views = {}
        self.processed_reports = set()
        self.user_report_counts = {}
        self.racist_classifier = RacistClassifier()
        self.flagged_messages = []

    def create_message_object(self, discord_message) -> Message:
        """Convert a Discord message to our Message class"""
        message = Message()
        message.content = discord_message.content or ""
        message.is_audio = len(discord_message.attachments) > 0 and any(
            attachment.content_type and attachment.content_type.startswith('audio/')
            for attachment in discord_message.attachments
        )

        if message.is_audio:
            # Find the first audio attachment
            audio_attachment = next(
                (att for att in discord_message.attachments
                 if att.content_type and att.content_type.startswith('audio/')),
                None
            )
            if audio_attachment:
                message.file_path = audio_attachment.url  # Discord CDN URL
                logger.info(f"[DEBUG] File path: {message.file_path}")
            else:
                message.file_path = None
        else:
            message.file_path = None

        return message

    async def classify_single_message(self, discord_message) -> bool:
        """Classify a single message and return True if racist content detected"""
        try:
            message_obj = self.create_message_object(discord_message)

            # Pass single message in a list
            is_racist = self.racist_classifier.make_prediction([message_obj])

            return is_racist

        except Exception as e:
            logger.error(f"Error classifying message: {str(e)}")
            return False

    async def classify_multiple_messages(self, discord_messages: List) -> bool:
        """Classify multiple messages together and return True if racist content detected"""
        try:
            message_objects = []

            for discord_message in discord_messages:
                message_obj = self.create_message_object(discord_message)
                message_objects.append(message_obj)

            is_racist = self.racist_classifier.make_prediction(message_objects)

            return is_racist

        except Exception as e:
            logger.error(f"Error classifying multiple messages: {str(e)}")
            return False

    async def handle_racist_content_detection(self, discord_message, is_reported=False):
        """Handle when racist content is detected"""
        try:
            if is_reported:
                # For reported messages, delete the message
                await discord_message.delete()
                logger.info(f"Deleted reported message from {discord_message.author.name} due to racist content detection")

                # Log the deletion
                deletion_info = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "message_id": discord_message.id,
                    "author": discord_message.author.name,
                    "author_id": discord_message.author.id,
                    "content": discord_message.content or "[Audio/Attachment]",
                    "channel": discord_message.channel.name,
                    "guild": discord_message.guild.name if discord_message.guild else "DM",
                    "reason": "Racist content detected in reported message"
                }

                # Send deletion notice to mod channel
                if discord_message.guild and discord_message.guild.id in self.mod_channels:
                    mod_channel = self.mod_channels[discord_message.guild.id]
                    embed = discord.Embed(
                        title="🚨 Reported Message Deleted - Racist Content Detected",
                        color=discord.Color.red(),
                        timestamp=datetime.datetime.now()
                    )
                    embed.add_field(
                        name="Message Details",
                        value=f"**Author:** {discord_message.author.name}\n**Content:** {discord_message.content or '[Audio/Attachment]'}\n**Channel:** {discord_message.channel.name}",
                        inline=False
                    )
                    await mod_channel.send(embed=embed)

            else:
                # For regular channel messages, flag for review
                flag_info = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "message_id": discord_message.id,
                    "author": discord_message.author.name,
                    "author_id": discord_message.author.id,
                    "content": discord_message.content or "[Audio/Attachment]",
                    "channel": discord_message.channel.name,
                    "guild": discord_message.guild.name if discord_message.guild else "DM",
                    "reason": "Racist content detected by classifier"
                }

                self.flagged_messages.append(flag_info)
                logger.info(f"Flagged message from {discord_message.author.name} for review due to racist content detection")

                # Send to mod channel for review
                if discord_message.guild and discord_message.guild.id in self.mod_channels:
                    mod_channel = self.mod_channels[discord_message.guild.id]
                    embed = discord.Embed(
                        title="⚠️ Message Flagged for Review - Racist Content Detected",
                        color=discord.Color.orange(),
                        timestamp=datetime.datetime.now()
                    )
                    embed.add_field(
                        name="Message Details",
                        value=f"**Author:** {discord_message.author.name}\n**Content:** {discord_message.content or '[Audio/Attachment]'}\n**Channel:** {discord_message.channel.name}\n**Jump to Message:** [Click here]({discord_message.jump_url})",
                        inline=False
                    )

                    # Add review buttons
                    view = MessageReviewView(discord_message, self)
                    flag_message = await mod_channel.send(embed=embed, view=view)
                    self.active_views[flag_message.id] = view

        except discord.errors.NotFound:
            logger.info("Message was already deleted")
        except discord.errors.Forbidden:
            logger.error("Bot doesn't have permission to delete messages")
        except Exception as e:
            logger.error(f"Error handling racist content detection: {str(e)}")

    async def get_context_messages(self, reported_message, context_count: int):
        """Get context messages before the reported message"""
        if context_count <= 0:
            return []

        try:
            # Get messages before the reported message
            messages = []
            async for message in reported_message.channel.history(
                limit=context_count + 1,
                before=reported_message.created_at
            ):
                messages.append(message)

            # Reverse to get chronological order (oldest first)
            messages.reverse()
            return messages

        except Exception as e:
            logger.error(f"Error getting context messages: {str(e)}")
            return []

    async def on_ready(self):
        print(f'{self.user.name} has connected to Discord! It is in these guilds:')
        for guild in self.guilds:
            print(f' - {guild.name}')
        print('Press Ctrl-C to quit.')

        # Parse the group number out of the bot's name
        match = re.search('[gG]roup (\d+) [bB]ot', self.user.name)
        if match:
            self.group_num = match.group(1)
        else:
            raise Exception("Group number not found in bot's name. Name format should be \"Group # Bot\".")

        # Find the mod channel in each guild that this bot should report to
        for guild in self.guilds:
            for channel in guild.text_channels:
                if channel.name == f'group-{self.group_num}-mod':
                    self.mod_channels[guild.id] = channel

    async def on_message(self, message):
        '''
        This function is called whenever a message is sent in a channel that the bot can see (including DMs).
        Currently the bot is configured to only handle messages that are sent over DMs or in your group's "group-#" channel.
        '''
        # Ignore messages from the bot
        if message.author.id == self.user.id:
            return

        # Debugging info
        if message.guild:
            logger.info(f"Channel message: {message.author.name} in {message.channel.name}: {message.content[:50]}")
        else:
            logger.info(f"DM from {message.author.name}: {message.content[:50]}")

        # Check if this message was sent in a server ("guild") or if it's a DM
        if message.guild:
            await self.handle_channel_message(message)
        else:
            await self.handle_dm(message)

    async def handle_dm(self, message):
        # Handle a help message
        if message.content.lower() == Report.HELP_KEYWORD:
            reply = "**Reporting Help**\n\n"
            reply += "Use the `report` command to begin the reporting process.\n"
            reply += "Use the `cancel` command to cancel the report process.\n"
            reply += "Follow the prompts and use the buttons/menus to complete your report."
            await message.channel.send(reply)
            return

        author_id = message.author.id

        # Only respond to messages in the reporting flow
        if author_id not in self.reports and not message.content.lower().startswith(Report.START_KEYWORD):
            return

        # Add active report for user
        if author_id not in self.reports:
            logger.info(f"Starting new report for {message.author.name}")
            self.reports[author_id] = Report(self)

        # Log the current state of the report before handling the message
        if author_id in self.reports:
            logger.info(f"Current report state for {message.author.name}: {self.reports[author_id].state}")

        # Let the report class handle this message; forward all the messages it returns to us
        responses = await self.reports[author_id].handle_message(message)

        # Log the new state of the report after handling the message
        if author_id in self.reports:
            logger.info(f"New report state for {message.author.name}: {self.reports[author_id].state}")

        for response in responses:
            # Check if the response is a tuple with message content and a view
            if isinstance(response, tuple) and len(response) == 2:
                content, view = response
                sent_message = await message.channel.send(content, view=view)
                logger.info(f"Sent message with view to {message.author.name}")

                # Store the view for interaction handling and set up the bot reference
                if view:
                    self.active_views[sent_message.id] = view
                    if hasattr(view, 'report'):
                        view.report.bot = self
            else:
                # Normal text message
                await message.channel.send(response)

        # If the report is complete, send it to the mod channel and remove it from our map
        if author_id in self.reports and self.reports[author_id].state == State.REPORT_COMPLETE:
            await self.process_completed_report(message.author, author_id)

    async def get_message_transcription(self, message_obj):
       """
       Get transcription for audio messages using the model's transcription capability
       """
       if message_obj.is_audio and message_obj.file_path:
           try:
               # This will call whisper when real model is implemented
               transcription = self.racist_classifier.mock_transcribe_audio(message_obj.file_path)
               return transcription
           except Exception as e:
               logger.error(f"Error transcribing audio: {str(e)}")
               return "[Transcription failed]"
       return None

    async def process_completed_report(self, author, author_id):
       """
       Process a completed report and send it to mod channel if not already sent
       """
       # Check if this report has already been processed
       report_id = f"{author_id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

       if report_id not in self.processed_reports:
           logger.info(f"Report is complete for {author.name}, processing with classifier")

           # Get the report object
           report = self.reports[author_id]

           # Increment prior reports
           if report.reported_user:
               user_id = report.reported_user.id
               self.user_report_counts[user_id] = self.user_report_counts.get(user_id, 0) + 1
               # Store in report object so it passes to embed
               report.prior_reports_count = self.user_report_counts[user_id]

           # Before sending, check for required data
           if not report.message:
               logger.error(f"Report has no message data: {report.get_report_data()}")
               await author.send("There was an issue with your report. Please try again or contact server administrators.")
           else:
               # Get context messages if requested
               messages_to_classify = [report.message]  # Start with the reported message

               if hasattr(report, 'context_messages') and report.context_messages > 0:
                   context_messages = await self.get_context_messages(report.message, report.context_messages)
                   # Add context messages before the reported message
                   messages_to_classify = context_messages + messages_to_classify

               # NEW LOGIC: Always classify ALL reported messages for hate speech detection
               try:
                   logger.info(f"Classifying {len(messages_to_classify)} message(s) for reported content from {author.name}")
                   is_racist = await self.classify_multiple_messages(messages_to_classify)

                   # If hate speech is detected, auto-delete the message regardless of report type
                   if is_racist:
                       logger.info(f"Hate speech detected in reported message from {author.name}, auto-deleting")
                       await self.handle_racist_content_detection(report.message, is_reported=True)
                       report.classifier_auto_deleted = True  # Mark for mod channel display
                   else:
                       logger.info(f"No hate speech detected in reported message from {author.name}")
                       report.classifier_auto_deleted = False

                   # Always send ALL reports to mod channel for human review
                   logger.info(f"Sending {report.report_type} report to mod channel for human review")
                   result = await self.send_report_to_mod_channel(author, report)

                   if result:
                       logger.info(f"Successfully sent {report.report_type} report to mod channel for {author.name}")
                       self.processed_reports.add(report_id)

                       # Notify the reporter with appropriate message
                       if is_racist:
                           await author.send("**Report Processed**\n\nThank you for your report. Our automated system detected policy violations and the message has been removed. The report has also been forwarded to our moderation team for additional review.")
                       else:
                           await author.send("**Report Submitted**\n\nThank you for your report. It has been forwarded to our moderation team for review.")
                   else:
                       logger.error(f"Failed to send {report.report_type} report to mod channel for {author.name}")
                       await author.send("Your report was submitted, but there might be an issue with the mod channel configuration. The administrators have been notified about this problem.")

               except Exception as e:
                   logger.error(f"Error processing report with classifier: {str(e)}")
                   # Fallback to sending to mod channel without classification
                   result = await self.send_report_to_mod_channel(author, report)
                   if result:
                       logger.info(f"Successfully sent {report.report_type} report to mod channel (fallback) for {author.name}")
                       self.processed_reports.add(report_id)
                   else:
                       logger.error(f"Failed to send {report.report_type} report to mod channel (fallback) for {author.name}")
                       await author.send("Your report was submitted, but there might be an issue with processing. The administrators have been notified.")

               # Store the data and remove the report
               self.completed_reports.append(report.get_report_data())
               self.reports.pop(author_id)

    async def handle_channel_message(self, message):
        # Only handle messages sent in the "group-#" channel
        if not message.channel.name == f'group-{self.group_num}':
            return

        # Classify the message for racist content
        try:
            logger.info(f"Classifying channel message from {message.author.name}")
            is_racist = await self.classify_single_message(message)

            if is_racist:
                logger.info(f"Racist content detected in channel message from {message.author.name}")
                await self.handle_racist_content_detection(message, is_reported=False)
            else:
                logger.info(f"No racist content detected in channel message from {message.author.name}")

        except Exception as e:
            logger.error(f"Error classifying channel message: {str(e)}")

    async def on_interaction(self, interaction):
        """
        Handle button clicks and select menu interactions
        """
        try:
            message_id = interaction.message.id
            user_id = interaction.user.id

            logger.info(f"Received interaction from {interaction.user.name} for message {message_id}")

            # Get the view associated with this message
            view = self.active_views.get(message_id)
            if not view:
                if interaction.message.ephemeral:
                    # Ignore ephemeral interaction messages
                    return
                logger.warning(f"No view found for message {message_id}")
                return

            logger.info(f"Found view for message {message_id}")

            # Check if this is a report view and the report is complete after interaction
            if hasattr(view, 'report') and hasattr(view.report, 'state'):
                if view.report.state == State.REPORT_COMPLETE and user_id in self.reports:
                    logger.info(f"Report is complete for {interaction.user.name} from interaction")
                    await self.process_completed_report(interaction.user, user_id)

        except discord.errors.HTTPException as e:
            # Handle interaction already acknowledged errors
            if "Interaction has already been acknowledged" in str(e):
                logger.info(f"Interaction already acknowledged: {str(e)}")
            else:
                logger.error(f"HTTP Exception in interaction: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in interaction handler: {str(e)}", exc_info=True)


    # Replace the send_report_to_mod_channel method in your ModBot class
    async def send_report_to_mod_channel(self, reporter, report):
        """
        Sends the completed report to the moderation channel with transcription if available
        Returns True if successful, False otherwise
        """
        try:
            if not report.message:
                logger.error("Cannot send report: No message was reported")
                return False

            guild_id = report.message.guild.id
            logger.info(f"Looking for mod channel in guild ID {guild_id}")

            if guild_id not in self.mod_channels:
                logger.error(f"Cannot send report: No mod channel found for guild ID {guild_id}")
                logger.error(f"Available guild IDs: {list(self.mod_channels.keys())}")
                return False

            mod_channel = self.mod_channels[guild_id]
            logger.info(f"Found mod channel: {mod_channel.name}")

            reporter_name = reporter.name if reporter else "Unknown"
            reported_name = report.reported_user.name if report.reported_user else "automated"
            message_content = report.message.content or "[No text content, message may be an audio file]"

            # Handle audio transcription
            transcription_text = ""
            if report.message.attachments:
                # Check if there are audio attachments and get transcription
                message_obj = self.create_message_object(report.message)
                if message_obj.is_audio:
                    transcription = await self.get_message_transcription(message_obj)
                    if transcription:
                        transcription_text = f"\n**Audio Transcription:**\n```{transcription}```\n"
                    else:
                        transcription_text = "\n**Audio File:** [Transcription unavailable]\n"
                else:
                    transcription_text = "\n**Attachments:** [Non-audio attachments present]\n"

            timestamp_str = report.message.created_at.strftime("%Y-%m-%d %H:%M:%S")
            targeted_status = "Yes" if report.hate_speech_targeted else "No"
            prior_reports = getattr(report, 'prior_reports_count', 0)
            context_messages = getattr(report, 'context_messages', 0)

            # Determine classifier result and auto-deletion status
            auto_deleted = getattr(report, 'classifier_auto_deleted', False)
            if auto_deleted:
                classifier_result = "HATE SPEECH DETECTED - Message automatically deleted"
                embed_color = discord.Color.red()
                title_prefix = "🚨 AUTO-DELETED"
            else:
                if report.report_type == Report.HATE_SPEECH:
                    classifier_result = "No hate speech detected (human review required)"
                else:
                    classifier_result = f"No hate speech detected (Non-hate speech report: {report.report_type})"
                embed_color = discord.Color.orange()
                title_prefix = "📋 PENDING REVIEW"

            embed = discord.Embed(
                title=f"{title_prefix} - Report: {report.hate_speech_type or report.report_type}",
                color=embed_color,
                timestamp=datetime.datetime.now()
            )

            # Add special notice for auto-deleted messages
            deletion_notice = ""
            if auto_deleted:
                deletion_notice = "⚠️ **MESSAGE WAS AUTOMATICALLY DELETED** - You can still take additional moderation actions below.\n\n"

            body = f"""
    {deletion_notice}**Reporter:** {reporter_name}
    **Reported User:** {reported_name}
    **Reported Message:**
    ```{message_content}```
    {transcription_text}
    **Timestamp:** {timestamp_str}
    **Prior Reports Against User:** {prior_reports}
    **Number of context messages requested:** {context_messages}
    **Targeted towards individual?** {targeted_status}
    **Classifier Result:** {classifier_result}
    """

            embed.add_field(name="Report Details", value=body, inline=False)

            # Create moderation view - this allows actions even on deleted messages
            view = ModerationReviewView(report, self)
            mod_message = await mod_channel.send(embed=embed, view=view)
            self.active_views[mod_message.id] = view

            logger.info("Report sent to mod channel with moderation view.")
            return True

        except Exception as e:
            logger.error(f"Error sending report to mod channel: {str(e)}")
            return False


class MessageReviewView(discord.ui.View):
    """View for reviewing flagged messages"""
    def __init__(self, flagged_message, bot):
        super().__init__(timeout=3600)  # 1 hour timeout
        self.flagged_message = flagged_message
        self.bot = bot

    @discord.ui.button(label="Delete Message", style=discord.ButtonStyle.danger)
    async def delete_message(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            await self.flagged_message.delete()
            await interaction.response.send_message("✅ Message deleted.", ephemeral=True)

            # Log the action
            logger.info(f"Moderator {interaction.user.name} deleted flagged message from {self.flagged_message.author.name}")

        except discord.errors.NotFound:
            await interaction.response.send_message("❌ Message was already deleted.", ephemeral=True)
        except discord.errors.Forbidden:
            await interaction.response.send_message("❌ Cannot delete message - insufficient permissions.", ephemeral=True)
        except Exception as e:
            await interaction.response.send_message(f"❌ Error deleting message: {str(e)}", ephemeral=True)

    @discord.ui.button(label="Keep Message", style=discord.ButtonStyle.success)
    async def keep_message(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message("✅ Message reviewed and kept.", ephemeral=True)
        logger.info(f"Moderator {interaction.user.name} kept flagged message from {self.flagged_message.author.name}")

    @discord.ui.button(label="Warn User", style=discord.ButtonStyle.secondary)
    async def warn_user(self, interaction: discord.Interaction, button: discord.ui.Button):
        # You could implement actual warning system here
        await interaction.response.send_message("✅ User warned (placeholder action).", ephemeral=True)
        logger.info(f"Moderator {interaction.user.name} warned user {self.flagged_message.author.name}")


client = ModBot()
client.run(discord_token)
