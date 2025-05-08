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
        self.mod_channels = {} # Map from guild to the mod channel id for that guild
        self.reports = {} # Map from user IDs to the state of their report
        self.completed_reports = [] # List to store completed reports
        self.active_views = {} # Map from message IDs to views
        self.processed_reports = set()  # Set to track reports that have been sent to mod channel

    async def on_ready(self):
        print(f'{self.user.name} has connected to Discord! It is these guilds:')
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
            reply =  "**Reporting Help**\n\n"
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

    async def process_completed_report(self, author, author_id):
        """
        Process a completed report and send it to mod channel if not already sent
        """
        # Check if this report has already been processed
        report_id = f"{author_id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

        if report_id not in self.processed_reports:
            logger.info(f"Report is complete for {author.name}, sending to mod channel")

            # Before sending, check for required data
            if not self.reports[author_id].message:
                logger.error(f"Report has no message data: {self.reports[author_id].get_report_data()}")
                await author.send("There was an issue with your report. Please try again or contact server administrators.")
            else:
                # Send to mod channel
                result = await self.send_report_to_mod_channel(author, self.reports[author_id])
                if result:
                    logger.info(f"Successfully sent report to mod channel for {author.name}")
                    self.processed_reports.add(report_id)  # Mark as processed
                else:
                    logger.error(f"Failed to send report to mod channel for {author.name}")
                    await author.send("Your report was submitted, but there might be an issue with the mod channel configuration. The administrators have been notified about this problem.")

            # Store the data and remove the report
            self.completed_reports.append(self.reports[author_id].get_report_data())
            self.reports.pop(author_id)

    async def handle_channel_message(self, message):
        # Only handle messages sent in the "group-#" channel
        if not message.channel.name == f'group-{self.group_num}':
            return

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

    async def send_report_to_mod_channel(self, reporter, report):
        """
        Sends the completed report to the moderation channel
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

#            await mod_channel.send("Report received")
            embed = discord.Embed(
                title="New Report Received",
                description=f"Reporter: {reporter.name}\nReported User: {report.reported_user.name if report.reported_user else 'Unknown'}\nType: {report.report_type}\nHate Speech Type: {report.hate_speech_type}",
                timestamp=datetime.datetime.now(),
                color=discord.Color.orange()
            )

            view = ModerationReviewView(report, self)

            await mod_channel.send(embed=embed, view=view)
            return True

        except Exception as e:
            logger.error(f"Error sending report to mod channel: {str(e)}")
            return False

client = ModBot()
client.run(discord_token)
