pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.PREFER_SETTINGS)
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()

        ivy {
            url = uri("https://nodejs.org/dist")
            patternLayout {
                artifact("v[revision]/[artifact]-v[revision]-[classifier].[ext]")
            }
            metadataSources {
                artifact()
            }
            content {
                includeGroup("org.nodejs")
            }
        }
    }
}

rootProject.name = "BA_social_media_overlay"
include(":app")
