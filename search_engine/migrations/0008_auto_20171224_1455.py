# -*- coding: utf-8 -*-
# Generated by Django 1.11.3 on 2017-12-24 14:55
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('search_engine', '0007_auto_20171130_0403'),
    ]

    operations = [
        #migrations.RunSQL(
            #('CREATE FULLTEXT INDEX document_index ON search_engine_documentdata (name, text)',),
            #('DROP INDEX document_index on search_engine_documentdata',)
        #),
    ]
